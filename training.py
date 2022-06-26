import argparse
import os
import torch

from tqdm import tqdm

import utils.logger as log_module
import utils.dataset as datasetutil
from models.trainable_wrapper import TrainableWrapper


if __name__ == "__main__":

    args = argparse.Namespace(
        log_path="log",
        dataset="imdb",
        gs=False,
        epochs=5,
        batch_size=64,
        lr=1e-4,
    )
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    logger = log_module.Logger(args.log_path)

    trainvalset, testset, (text_key, testset_key) = datasetutil.get_dataset(args)

    trainset, valset = datasetutil.split_dataset(trainvalset, split="trainval", split_ratio=0.8)

    wrapper = TrainableWrapper(args, logger)
    optimizer = torch.optim.Adam(wrapper.parameters(), args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_train_samples = len(trainset["label"])
    num_train_batches = int((num_train_samples // args.batch_size) + 1)
    
    num_val_samples = len(valset["label"])
    num_val_batches = int((num_val_samples // args.batch_size) + 1)

    for e in range(args.epochs):
        print(f"Epoch {e + 1}")
        wrapper.train(True)
        for batch in tqdm(range(num_train_batches)):
            lower = batch * args.batch_size
            upper = min((batch + 1) * args.batch_size, num_train_samples)
            examples = trainset[text_key][lower:upper]
            if len(examples) == 0:
                continue
            labels = trainset["label"][lower:upper]
            labels_tensor = torch.LongTensor(labels).to("cuda")
            optimizer.zero_grad()

            out = wrapper(examples)
            loss = loss_fn(out, labels_tensor)
            loss.backward()
            optimizer.step()


        wrapper.train(False)
        with torch.no_grad():
            accu_loss = 0.0
            correct = 0
            for batch in tqdm(range(num_val_batches)):
                lower = batch * args.batch_size
                upper = min((batch + 1) * args.batch_size, num_val_samples)
                examples = trainset[text_key][lower:upper]
                if len(examples) == 0:
                    continue
                labels = trainset["label"][lower:upper]
                labels_tensor = torch.LongTensor(labels).to("cuda")
                out = wrapper(examples)

                loss = loss_fn(out, labels_tensor)
                accu_loss += float(loss.cpu())
                
                pred = out.argmax(1)
                correct += (pred == labels_tensor).sum().cpu().numpy()
            
            print("Validation:")
            print(f"{accu_loss} / {num_val_samples} = {accu_loss / num_val_samples:0.10f}")
            print(f"{correct} / {num_val_samples} = {correct / num_val_samples:0.10f}")

    wrapper.train(False)
    with torch.no_grad():
        accu_loss = 0.0
        correct = 0
        for batch in tqdm(range(num_val_batches)):
            lower = batch * args.batch_size
            upper = min((batch + 1) * args.batch_size, num_val_samples)
            examples = testset["text"][lower:upper]
            if len(examples) == 0:
                continue
            labels = testset["label"][lower:upper]
            labels_tensor = torch.LongTensor(labels).to("cuda")
            out = wrapper(examples)

            loss = loss_fn(out, labels_tensor)
            accu_loss += float(loss.cpu())
            
            pred = out.argmax(1)
            correct += (pred == labels_tensor).sum().cpu().numpy()

        print("Test result:")
        print(f"{accu_loss} / {num_val_samples} = {accu_loss / num_val_samples:0.10f}")
        print(f"{correct} / {num_val_samples} = {correct / num_val_samples:0.10f}")





