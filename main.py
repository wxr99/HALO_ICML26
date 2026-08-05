# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
import copy
import argparse
import yaml
import logging
import os
import random
import numpy as np
from typing import Dict, Any
from dataset.reformat_tree import reformat_tree
from memory.reservoir import ReservoirMemory
from memory.plfms import PLFMSMemory
from memory.cbrs import CBRSMemory
from memory.clib import ClibMemory
from memory.dhbrs import DHBMemory
from sklearn.metrics import auc
import clip
from datetime import datetime

# --- WandB Import ---
try:
    import wandb
    # import swanlab as wandb  # 使用 swanlab 替代 wandb
except ImportError:
    print("wandb not installed. Please install with 'pip install wandb'")
    wandb = None

# wandb.init(settings=wandb.Settings(silent=True))


# --- Import Custom Modules ---
try:
    from networks.heads_setup import configure_heads_from_config
    from networks.resnet import ResNetMultiHeadHierarchical, clone_resnet_snapshot
    from networks.vit import ViTMultiHeadHierarchical, CustomDeiT
    from dataset.dataloader import get_dataloader
    from train import train_epoch         # 导入训练函数
    from train_clip import train_epoch_clip  
    from eval import evaluate_model  # 导入评估函数
    from eval_clip import evaluate_model_clip
    from utils.label_mapper import seen_classes_per_level
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure models.py, data_utils.py, train.py, evaluate.py, and utils.py exist.")
    exit(1)

# --- Setup Logging (Python Standard Logger) ---
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configures logging to file and console."""
    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    log_dir = config['output_dir']
    os.makedirs(log_dir, exist_ok=True)
    # 使用 run_name 确保日志文件名唯一
    run_name = config.get("run_name", "run")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{run_name}_{current_time}.log")

    # 移除旧的 handlers 防止重复日志
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    # logger.info(f"Logging initialized. Log file: {log_filename}") # 在 wandb 初始化后记录
    return logger

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Set Random Seeds ---
def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Main Orchestration Function ---
def main(args):
    # --- Load Config & Override ---
    config = load_config(args.config)
    config['epochs'] = args.epochs if args.epochs is not None else config.get('epochs', 10)
    config['output_dir'] = args.output_dir if args.output_dir else config.get('output_dir', './output_default')
    config['run_name'] = args.run_name if args.run_name else config.get('run_name', 'default_run')
    config['output_dir'] = os.path.join(config['output_dir'], config['run_name']) # Specific run directory
    config['wandb_project'] = args.wandb_project if args.wandb_project else config.get('wandb_project', 'default_project')
    config['wandb_entity'] = args.wandb_entity if args.wandb_entity else config.get('wandb_entity', None)
    eval_interval = config.get('eval_interval', 100)  # 每隔多少次训练进行一次评估
    if args.no_wandb:
        wandb_mode = "disabled"
    else:
        wandb_mode = config.get('wandb_mode', 'online')

    # --- Setup Logging (Python Logger) ---
    logger = setup_logging(config)
    logger.info("Starting main script...")
    logger.info(f"Run Name: {config['run_name']}")
    logger.info(f"Output Directory: {config['output_dir']}")

    # --- Initialize WandB ---
    current_wandb_run = None # 初始化为 None
    if wandb and wandb_mode != "disabled":
        try:
            current_wandb_run = wandb.init(
                project=config['wandb_project'],
                entity=config['wandb_entity'],
                config=config,
                name=config['run_name'],
                dir=config['output_dir'],
                mode=wandb_mode,
                resume="allow", # 允许恢复中断的运行 (可选)
                id=wandb.util.generate_id() if not args.resume_wandb_id else args.resume_wandb_id # 生成新ID或使用提供的ID恢复
            )
            logger.info(f"WandB initialized successfully. Mode: {wandb_mode}. Run URL: {current_wandb_run.get_url()}")
            # 保存 YAML 配置到 WandB
            wandb.save(args.config)
            # 记录最终配置到日志
            logger.info(f"Full configuration logged to WandB: \n{yaml.dump(wandb.config, indent=2)}")

        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
            logger.warning("Proceeding without WandB logging.")
    else:
        logger.info("WandB is disabled.")

    # --- Setup: Seed, Device ---
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Main Execution Block with Error Handling ---
    try:
        error_occurred = False
        # --- Create Dataloaders ---
        # 注意：get_dataloader 现在需要能处理真实的 dataset 对象
        try:
            dataloaders, label_tree = get_dataloader(
                name=config['dataset'],
                num_tasks=config['num_tasks'], 
                overlap_ratio=config['overlap_ratio'],                       
                val_split=config['split_ratio'],                
                batch_size=config['batch_size'],
                num_workers=config.get('num_workers', 4)
            )
        except Exception as e:
             logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
             raise # 重新抛出异常，因为没有数据无法继续

        # --- Initialize Model ---
        add_heads = config.get("add_heads", {})
        # print(add_heads)
        head_list = config.get('head_type', None)  # 获取头部类型列表
        logger.info(f"Initializing model: {config['backbone_name']}")
        try:
            if config['backbone_name'] == 'resnet50':
                model = ResNetMultiHeadHierarchical(
                    backbone_name=config['backbone_name'],
                    custom_weight_path=config.get('pretrained_path', None),
                    pretrained=True if config.get('pretrained_path') else True,
                    return_features=config.get('return_features', False),
                    freeze_backbone=config.get('freeze_backbone', False),
                    expansion_dim=config.get('expansion_dim', 4096),
                    proto_dim=config.get('proto_dim', 256),
                    add_heads=add_heads,
                    head_list=head_list,
                )
            elif config['backbone_name'] == 'vit':
                model = ViTMultiHeadHierarchical(
                    backbone_ctor=CustomDeiT,
                    backbone_kwargs=dict(
                    img_size=config.get('img_size', 224),
                    patch_size=config.get('patch_size', 16),
                    in_chans=config.get('in_chans', 3),
                    num_classes=0,
                    embed_dim=config.get('embed_dim', 768),
                    depth=config.get('depth', 12),
                    num_heads=config.get('num_heads', 12),
                    mlp_ratio=config.get('mlp_ratio', 4.0),
                    qkv_bias=True,
                    drop_rate=config.get('drop_rate', 0.0),
                    attn_drop_rate=config.get('attn_drop_rate', 0.0),
                    drop_path_rate=config.get('drop_path_rate', 0.0),
                    has_cls_token=True
                    ),
                    custom_weight_path=config.get('pretrained_path', None),
                    pretrained=True if config.get('pretrained_path') else True,
                    return_features=config.get('return_features', False),
                    freeze_backbone=config.get('freeze_backbone', False),
                    expansion_dim=config.get('expansion_dim', 4096), # 若你的 ViT 头里会用到
                    proto_dim=config.get('proto_dim', 256),
                    add_heads=add_heads, # 与 resnet 相同：是否在构造时就添加 heads
                    head_list=head_list # 与 resnet 相同：[(name, num_classes, head_type, source), ...]
                    )
            elif config['backbone_name'] == 'clip':
                model, preprocess = clip.load("ViT-B/16", device=device, download_root="")
                logger.info(f"CLIP model initialized with architecture: ViT-B/16")
                custom_model_path = config.get('pretrained_path', None)
                if custom_model_path:
                    logger.info(f"Loading custom weights from {custom_model_path}...")
                    state_dict = torch.load(custom_model_path, map_location=device)  # 加载本地权重
                    model.load_state_dict(state_dict)
                    logger.info("Local weights loaded successfully!")
                else:
                    logger.info("No custom CLIP weights provided, using default weights.")
            
            added_heads_names = configure_heads_from_config(
                model=model,
                config=config,
                logger=logger,
                head_list=head_list  # 这里传你已有的 head_list；没有就传 None
            )

            if config.get('loss_function') == 'hieproloss':
                model_init = clone_resnet_snapshot(model).to(device)
                model_init.eval()  # 设置为评估模式，冻结参数
            else:
                model_init = None

            added_heads_names = list(dict.fromkeys(added_heads_names))
            
            model.to(device)
            logger.info("Model initialized and moved to device.")
            # 确定主头用于检查点
            primary_head = config['primary_head'] if added_heads_names else None
            if primary_head: logger.info(f"Using '{primary_head}' for primary validation metric.")

            

        except Exception as e:
             logger.error(f"Failed to initialize model: {e}", exc_info=True)
             raise

        # --- Loss Criterion ---
        # criterion = nn.CrossEntropyLoss(**criterion_args).to(device)
        loss_function = config['loss_function']
        logger.info(f"Loss criterion: {loss_function}")

        # --- WandB Watch Model ---
        if current_wandb_run:
            logger.info("Setting up wandb.watch...")
            # 确保在优化器创建 *之后* 调用 watch，如果需要记录梯度
            # 但如果只记录模型结构，可以在此调用
            current_wandb_run.watch(model, loss_function, log="all", log_freq=config.get('wandb_log_freq', 100))

        # --- Optimizer ---
        optimizer_name = config.get('optimizer', 'Adam').lower()
        lr = config.get('learning_rate', 0.001)
        wd = config.get('weight_decay', 0.0)
        # 确保只优化需要训练的参数
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_trainable}")

        if optimizer_name == 'adam':
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(trainable_params, lr=lr, momentum=config.get('sgd_momentum', 0.9), weight_decay=wd)
        # 可选：添加 AdamW
        elif optimizer_name == 'adamw':
             optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        logger.info(f"Optimizer: {optimizer_name.capitalize()}, LR: {lr}, Weight Decay: {wd}")

        # 可选：学习率调度器
        scheduler = None
        scheduler_name = config.get('scheduler', None)
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=config.get('lr_step_size', 30),
                                                  gamma=config.get('lr_gamma', 0.1))
            logger.info(f"Using StepLR scheduler: step_size={config.get('lr_step_size', 30)}, gamma={config.get('lr_gamma', 0.1)}")
        elif scheduler_name == 'cosine':
             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=config['epochs'], # 通常设置为总周期数
                                                              eta_min=config.get('lr_eta_min', 0))
             logger.info(f"Using CosineAnnealingLR scheduler: T_max={config['epochs']}, eta_min={config.get('lr_eta_min', 0)}")
        # 添加其他调度器...


        # --- Training Loop ---
        logger.info("Starting training loop...")
        global_batch = 0 
        eval_results = []
        start_epoch = 0 # 可用于恢复训练

        memory_name = config.get('memory_name', 'Reservoir')
        memory_capacity = config.get('memory_capacity', 1000)  # 可从配置中读取
        num_classes = config.get('num_classes', 10)  # 可从配置中读取
        if memory_name == 'Reservoir':
            memory = ReservoirMemory(max_size=memory_capacity)
        elif memory_name == 'PLFMS':
            memory = PLFMSMemory(max_size=memory_capacity)
        elif memory_name == 'Clib':
            memory = ClibMemory(max_size=memory_capacity)
        elif memory_name == 'CBRS':
            memory = CBRSMemory(max_size=memory_capacity)
        elif memory_name == 'DHBRS':
            memory = DHBMemory(max_size=memory_capacity)
        else:
            memory = None 

        if config['loss_function'] == 'hieproloss':
            mem_val = DHBMemory(max_size=config.get('mem_val_capacity', 200))  # 可从配置中读取)
        else:
            mem_val = None

        seen_classes = []  # 用于跟踪已见类别
        for task_id, task_loader in enumerate(dataloaders):
            train_loader = task_loader['train_loader']
            val_loader = task_loader['val_loader']
            test_loader = task_loader['test_loader']
            # --- 训练和评估 ---
            logger.info(f"Training on task {task_id + 1}/{len(dataloaders)}")
            logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
            for epoch in range(start_epoch, config['epochs']):
                if task_id == 0:
                    first_task = True
                    model_old = None
                    old_class_indexes = None
                else:
                    first_task = False
                if task_id == 1:
                    second_task = True
                else:
                    second_task = False
                # 计算当前周期的起始全局步数
                global_step_base = epoch * len(train_loader)
                if config['backbone_name'] == 'clip':
                    if config['post_train']:
                        global_batch += len(train_loader)

                    else:
                        train_loss, seen_classes = train_epoch_clip(
                            model=model, head_names=added_heads_names, dataloader=train_loader, test_loader=test_loader,
                            global_label_tree=label_tree, seen_classes=seen_classes, criterion=loss_function,optimizer=optimizer, device=device, 
                            epoch=epoch, global_batch=global_batch, logger=logger, memory=memory,
                            config=config, global_step_base=global_step_base, eval_interval=eval_interval, 
                            eval_results=eval_results, wandb_run=current_wandb_run, primary_head=primary_head
                        )

                        global_batch += len(train_loader)

                        # val_loss, val_accuracies = evaluate_model_clip(
                        #     model=model, head_names=added_heads_names, primary_head=primary_head, dataloader=val_loader,
                        #     global_label_tree=label_tree, current_label_tree=label_tree, seen_classes=seen_classes, device=device, logger=logger, config=config, epoch=None, 
                        #     prefix="val", wandb_run=current_wandb_run
                        # )
                else:
                    # 调用训练函数
                    train_loss, seen_classes = train_epoch(
                        model=model, model_old=model_old, model_init= model_init, head_names=added_heads_names, dataloader=train_loader, 
                        test_loader=test_loader, global_label_tree=label_tree, seen_classes=seen_classes, criterion=loss_function,optimizer=optimizer,
                        device=device, epoch=epoch, global_batch=global_batch, logger=logger, memory=memory,
                        config=config, global_step_base=global_step_base, eval_interval=eval_interval, 
                        eval_results=eval_results, wandb_run=current_wandb_run, first_task=first_task, second_task=second_task, 
                        old_class_indexes=old_class_indexes, mem_val=None
                    )

                    global_batch += len(train_loader)

                    # 调用评估函数进行验证
                    # val_loss, val_accuracies = evaluate_model(
                    #     model=model, head_names=added_heads_names, primary_head=primary_head, dataloader=val_loader, criterion=loss_function,
                    #     global_label_tree=label_tree, current_label_tree=None, seen_classes=seen_classes, device=device, logger=logger, config=config, epoch=epoch,
                    #     prefix="val", wandb_run=current_wandb_run
                    # )

                # 更新学习率调度器 (如果使用)
                if scheduler:
                    scheduler.step()
                    if current_wandb_run: # 记录当前学习率
                        current_lr = optimizer.param_groups[0]['lr']
                        current_wandb_run.log({"learning_rate": current_lr}, step=global_step_base + len(train_loader))

            old_class_indexes = seen_classes_per_level(seen_classes=seen_classes)

            logger.info(f"Task {task_id+1} Training finished.")

            # --- Final Evaluation on Test Set ---
            logger.info("Evaluating on Test Set using the current model...")
            model_final = model # 使用训练结束时的模型
            
            if config['backbone_name'] == 'resnet50':
                model_old = clone_resnet_snapshot(model).to(device)

            if config['backbone_name'] == 'clip':
                logger.info("Testing the zero-shot performance of CLIP model...")
                test_loss, test_accuracies = evaluate_model_clip(
                    model=model_final, head_names=added_heads_names, primary_head=primary_head, dataloader=test_loader,
                    global_label_tree=label_tree, current_label_tree=label_tree, seen_classes=seen_classes, device=device, logger=logger, config=config, epoch=None, 
                    prefix="test", wandb_run=current_wandb_run
                )

            else:
                # 调用评估函数进行测试
                test_loss, test_accuracies = evaluate_model(
                    model=model_final, model_init=model_init, head_names=added_heads_names, primary_head=primary_head, dataloader=test_loader,
                    global_label_tree=label_tree, current_label_tree=label_tree, seen_classes=seen_classes, criterion=loss_function, device=device, logger=logger, config=config, epoch=None, 
                    prefix="test", wandb_run=current_wandb_run
                )
            # --- 记录最终测试指标到 WandB 摘要 ---
            if current_wandb_run:
                current_wandb_run.summary["final_test_loss"] = test_loss
                for head_name, acc in test_accuracies.items():
                    current_wandb_run.summary[f"final_test_accuracy_{head_name}"] = acc
                # current_wandb_run.summary["final_test_accuracy"] = test_accuracies
            
    except Exception as e:
        logger.error("An critical error occurred during the main execution:", exc_info=True)
        error_occurred = True # Set the flag indicating an error
    # No need to call finish or mark_failed here, finally block will handle it

    finally:
    # --- Finish WandB Run ---

        # 初始化存储指标的字典
        metrics_data = {}

        total_batches = max([step for step, _ in eval_results])

        # 遍历 eval_results，收集所有指标的数据
        for step, metrics in eval_results:
            for key, value in metrics.items():
                if key not in metrics_data:
                    metrics_data[key] = {"x": [], "y": []}  # 初始化存储结构
                metrics_data[key]["x"].append(step)  # 添加全局步数
                metrics_data[key]["y"].append(value)  # 添加对应指标值

        # 计算所有指标的 AUC
        all_metrics_auc = {}
        for key, data in metrics_data.items():
            if len(data["x"]) > 1:  # 至少需要两个点才能计算 AUC
                all_metrics_auc[key] = auc(data["x"], data["y"])
            else:
                all_metrics_auc[key] = None  # 如果数据不足，标记为 None

        # 打印和记录 AUC
        for key, auc_score in all_metrics_auc.items():
            if auc_score is not None:
                logger.info(f"AUC for {key}: {auc_score/total_batches:.4f}")
                if current_wandb_run:
                    current_wandb_run.log({f"{key}_auc": auc_score/total_batches})
            else:
                logger.warning(f"Insufficient data to calculate AUC for {key}.")

        if current_wandb_run:
            # Determine exit code based on whether an error occurred
            exit_code = 1 if error_occurred else 0
            logger.info(f"Finishing WandB run... (Exit code: {exit_code})")
            try:
                current_wandb_run.finish(exit_code=exit_code)
            except Exception as wb_finish_err:
                logger.error(f"Error during wandb.finish: {wb_finish_err}", exc_info=True)
        logger.info("Script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Multi-Head Model Training Orchestrator')
    parser.add_argument('--config', type=str, default='./config/aliasing/cub/resnet/online/softce.yaml',
                        help='Path to the configuration YAML file (default: template.yaml)')
    # Overrides
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--run-name', type=str, default=None, help='Override run name for logging and wandb')
    # WandB specific args
    parser.add_argument('--wandb-project', type=str, default=None, help='Override wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Override wandb entity (username/team)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--resume-wandb-id', type=str, default=None, help='WandB run ID to resume (optional)')
    # 可选：恢复检查点
    # parser.add_argument('--resume-checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from (optional)')


    args = parser.parse_args()

    # --- 参数验证 ---
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    if not wandb and not args.no_wandb:
        print("Warning: wandb library not found, but --no-wandb flag was not set. Disabling wandb.")
        args.no_wandb = True # 强制禁用

    main(args)