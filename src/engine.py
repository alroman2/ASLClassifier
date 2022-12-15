from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch
import matplotlib.pyplot as plt
import time

# global settings
plt.style.use('ggplot')


def train(train_data_loader, model):
    print("Training started...")
    global train_itr
    global train_loss_list

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()

        train_itr += 1

        prog_bar.set_description(desc=f'loss={loss_value:.4f}')

    return train_loss_list


def validate(validate_data_loader, model):
    print("validation started...")
    global val_itr
    global val_loss_list

    prog_bar = tqdm(validate_data_loader, total=len(validate_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)

        val_itr += 1

        prog_bar.set_description(desc=f'loss={loss_value:.4f}')
        return val_loss_list


if __name__ == '__main__':
    # init model and move to gpu or cpu
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # get mode params
    params = [p for p in model.parameters() if p.requires_grad]
    # init optimizer
    optimizer = torch.optim.SGD(
        params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_loss_hist = Averager()
    val_loss_hist = Averager()

    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []

    MODEL_NAME = 'ASL-DETECTION-RESNET50'

    if VISUALIZE_TRANSORMED_IMAGES:
        from utils import show_transformed_image
        show_transformed_image(train_loader)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}/{NUM_EPOCHS}')

        # reset the training and validation loss histories for this epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # create two sublots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, val_ax = plt.subplots()

        # start time and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} val loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end-start)/60):.3f} minutes for epech #{epoch}")

        if (epoch+1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(),
                       f'{OUT_DIR}/{MODEL_NAME}_{epoch+1}.pth')
            print(f"Model saved at {OUT_DIR}/{MODEL_NAME}.pth")

        if (epoch+1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('Iteration')
            train_ax.set_ylabel('Train Loss')
            val_ax.plot(val_loss, color='red')
            val_ax.set_xlabel('Iteration')
            val_ax.set_ylabel('Validation Loss')
            figure_1.savefig(f'{OUT_DIR}/train_loss_{epoch+1}.png')
            figure_2.savefig(f'{OUT_DIR}/val_loss_{epoch+1}.png')
            print(
                f"Plots saved at {OUT_DIR}/train_loss_{epoch+1}.png and {OUT_DIR}/val_loss_{epoch+1}.png")

        if (epoch+1) == NUM_EPOCHS:  # save the final model
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('Iteration')
            train_ax.set_ylabel('Train Loss')
            val_ax.plot(val_loss, color='red')
            val_ax.set_xlabel('Iteration')
            val_ax.set_ylabel('Validation Loss')
            figure_1.savefig(f'{OUT_DIR}/train_loss_{epoch+1}.png')
            figure_2.savefig(f'{OUT_DIR}/val_loss_{epoch+1}.png')
            torch.save(model.state_dict(),
                       f'{OUT_DIR}/{MODEL_NAME}_{epoch+1}.pth')

        plt.close('all')
