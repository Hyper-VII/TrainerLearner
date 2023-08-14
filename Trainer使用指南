# Trainer 使用指南

本文以huggingface/transformers库的run_image_classification.py例程为例，介绍使用Trainer进行深度学习模型测试/训练。

参考：
[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

[HuggingFace 使用load_dataset读取数据集](https://zhuanlan.zhihu.com/p/634098463)

使用Trainer进行训练，与Datasets库进行配合使用，十分方便。
强烈建议在使用Datasets库读取数据，配合Trainer使用。
数据读取脚本教程可以阅读参考链接获取。

## Trainer脚本的主体结构

``` python

@dataclass
class DataTrainingArguments:
    """ 数据相关的参数设置。 """

@dataclass
class ModelArguments:
    """ 预训练模型的相关参数。 """

def collate_fn(examples :dict):
    """ 数据整理函数,在定义Trainer变量时会用到 """

def main():
    """ 程序主体 """

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 利用Datasets库数据读取脚本定义数据集
    dataset = load_dataset(data_args.dataset_name,)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True,
    )

    # 定义trainer变量
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics_func,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # 训练过程
    if training_args.do_train:
        pass
    # 评价过程
    if training_args.do_eval:
        pass

if __name__ == "__main__":
    main()
```

## 参数的设置

Trainer自身包含丰富的参数供训练/测试过程中使用，一般情况下，可以到源码的training_args.py文件中查看，设置对应参数的值。
当需要增加或者修改其中的某些参数，可以定义数据类进行参数的设置。

从上节中可以看到，DataTrainingArguments类和ModelArguments类都是经由dataclass装饰的数据类，用来进行参数的设置。

## collate_fn函数的意义

collate_fn函数是十分必要的，用来将单个的数据按照batch进行归类返回。

以图像分类的示例来举例，我们假设数据读取脚本返回的数据是{'pixel_values':<np.ndarray>,'labels':<int>}格式的dict类型——在Dataset数据读取脚本的编写中也应该遵循这一规范，则传入collate_fn函数的参数是长度为batch_size的list类型，而list的每一项即为上边的dict形式，也就是在编写数据读取脚本时候，_generate_examples函数的返回值。

而collate_fn函数的作用是将[{'pixel_values':<np.ndarray>,'labels':<int>},...]形式的数据转换为{'pixel_values':list[<np.ndarray>],'labels':list[<int>]}。

在定义trainer的时候，如果提供了data_collator参数，则tokenizer参数不是必须的，会优先使用data_collator提供的Callable对象处理数据。

## 函数主体

首先是参数的读取。

``` dataset = load_dataset(data_args.dataset_name,) ```

这里的dataset_name参数传入写好的数据读取脚本{xxx.py}

__config__ :  定义config时候传入的四个参数分别是模型的路径、labels的个数(num_labels)、{label(str):id(int)}形式的字典(label2id)、{id(int):label(str)}形式的字典(id2label)。

__model__ : 定义model的时候同样也需要传入模型的路径，还有刚刚定义的config变量。同时建议，将ignore_mismatched_sizes=True加上，以免在finetune模型时候出现分类头参数形状不符，而报错。

__trainer__ : 定义trainer时，传入定义好的模型和args训练参数项，具体参数值可以查阅training_args.py中获取，大多数使用默认值就可以了。tokenizer和data_collator参数只需要传入一个，data_collator的优先级较高，会屏蔽tokenizer参数传入的对象作用。

在脚本中定义了do_train do_eval do_predict几个参数，但这几个参数程序并不使用，而是提供给编码人员进行不同条件下的程序控制。因此，我们在写后面的流程时可以将训练流程、测试流程分开，分别编码。

当调用trainer.train()时，自动按照设定的参数和模型开始训练。

下一篇文章将会对Trainer训练运行过程进行说明，敬请期待。
