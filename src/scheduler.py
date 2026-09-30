from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def get_scheduler(config, optimizer, train_loader_length):
    train_steps = train_loader_length * config.epochs
    warmup_steps = int(train_loader_length * config.warmup_epochs)
    print("Warmup steps:", warmup_steps)
    print("Train steps: ", train_steps)

    if config.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    if config.scheduler == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    if config.scheduler == "constant":
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    return None
