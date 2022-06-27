import argparse
import os
import numpy as np
import torch

from tqdm import tqdm

import utils.logger as log_module
import utils.dataset as datasetutil
from models.trainable_wrapper import TrainableWrapper

from Detector import Detector
from AttackLoader import AttackLoader

def evaluate(model, data, labels, batch_size=64, loss_fn=torch.nn.CrossEntropyLoss()):
    num_samples = len(labels)
    num_batches = int((num_samples // batch_size) + 1)
    model.train(False)
    with torch.no_grad():
        accu_loss = 0.0
        correct = 0
        for batch in tqdm(range(num_batches)):
            lower = batch * batch_size
            upper = min((batch + 1) * batch_size, num_samples)
            data_batch = data[lower:upper]
            if len(data_batch) == 0:
                continue
            labels_batch = labels[lower:upper]
            labels_batch_tensor = torch.LongTensor(labels_batch).to("cuda")
            out = model(data_batch)

            loss = loss_fn(out, labels_batch_tensor)
            accu_loss += float(loss.cpu())
            
            pred = out.argmax(1)
            correct += (pred == labels_batch_tensor).sum().cpu().numpy()
        print(f"{accu_loss} / {num_samples} = {accu_loss / num_samples:0.10f}")
        print(f"{correct} / {num_samples} = {correct / num_samples:0.10f}")
    return accu_loss / num_samples, correct / num_samples



if __name__ == "__main__":

    args = argparse.Namespace(
        log_path="log",
        dataset="imdb",
        gs=False,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        # AttackLoader
        data_root_dir="attack-log/strong",
        data_type="standard",
        scenario="s1",
        model_type="bert",
        attack_type="pwws", # "textfooler"
        include_fae=False,
        unbalanced=False,
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

        print("Validation:")
        evaluate(model=wrapper, data=valset[text_key], labels=valset["label"], batch_size=args.batch_size)


    print("Test:")
    evaluate(model=wrapper, data=testset[text_key], labels=testset["label"], batch_size=args.batch_size)

    loader = AttackLoader(args, logger, data_type=args.data_type)
    atck_test, dataset = loader.get_attack_from_csv(batch_size=args.batch_size)
    atck_test = atck_test[~atck_test["perturbed_text"].isna()]
    print("Attack:")
    evaluate(model=wrapper, data=atck_test["perturbed_text"].tolist(), labels=atck_test["ground_truth_output"].tolist(), batch_size=args.batch_size)