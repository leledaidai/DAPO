### 本仓库的DAPO算法是由[[om-ai-lab/VLM-R1: Solve Visual Understanding with Reinforced VLMs](https://github.com/om-ai-lab/VLM-R1/tree/main?tab=readme-ov-file)]中的GRPO算法改写得到

### 算法在vrl框架上实现，相比于原有代码主要修改如下部分：

- 添加`dapo_trainer.py`，构造`VLMDAPOTrainer`类，修改`VLMGRPOTrainer`类中的`compute_loss`等函数，添加`_get_train_sampler`、`get_sample_dataloader`等函数
- 修改`vlm_modules`中的`qwen_module.py`，增加`length_reward`函数
- 对vrl框架中的参数设置进行适配
- 实现了DAPO论文中提及的全部改进，即：
  - ![001](./assets/001.png)
  - ![002](./assets/002.png)
  - ![004](./assets/004.png)
  - ![003](./assets/003.png)

