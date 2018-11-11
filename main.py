import numpy as np
from common_blocks import utils, models, loaders
import os
from torch import optim, nn
import copy
import torch
import pandas as pd

# global param
data_path = './data/input/'
output_path = './data/output/'
model_save_path = os.path.join(output_path, 'cache/resnet152.pth')
BATCH_SIZE = 16
EPOCHS = 99999
DEVICE = torch.device('cuda:0')
logger = utils.get_logger('tiangong')


def train(model, dataloaders, loss_fn, update_layer, train_on_val=False, lr=1e-4):
    # train specific layer
    for param in update_layer.parameters():
        param.requires_grad = True

    # train
    model = model.to(DEVICE)
    dataloader_train, dataloader_val = dataloaders
    # todo: change to detect params
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = optim.Adam(params_to_update, lr=lr)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0
    for epoch in range(EPOCHS):
        # train
        model.train()
        running_loss = 0.0
        running_correct = 0
        num = 0
        for images, labels in dataloader_train:
            batch_size = images.size(0)
            num += batch_size
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                _, labels_ = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_size
                running_correct += torch.sum(labels_ == labels.data)
        epoch_loss = running_loss / num
        epoch_acc = running_correct.double() / num

        # eval
        if not train_on_val:
            epoch_loss_val, epoch_acc_val = predict_eval(model=model,
                                                         dataloader=dataloader_val,
                                                         loss_fn=loss_fn)
            if epoch_acc_val > best_acc:
                patience = 0
                best_acc = epoch_acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= 50:
                    logger.info('early stop')
                    break
            logger.info('train loss: {:.4f} acc: {:.4f}; val loss: {:.4f} acc: {:.4f}'.format(
                epoch_loss, epoch_acc, epoch_loss_val, epoch_acc_val))
        else:
            raise NotImplementedError
    # save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)
    logger.info('saved model. best acc:{}'.format(best_acc))
    return model


def predict_eval(model, dataloader, loss_fn):
    labels_pred = []
    model.eval()
    running_loss = 0.0
    num = 0
    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.size(0)
            num += batch_size
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.set_grad_enabled(False):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                _, labels_ = torch.max(outputs, 1)
                running_loss += loss.item() * batch_size
                labels_pred.append(labels_.cpu().numpy())
    loss = running_loss / num
    labels_pred = np.concatenate(labels_pred, axis=0)
    acc = sum(labels_pred == dataloader.dataset.labels)/labels_pred.shape[0]
    return loss, acc, labels_pred


# data
metadata = loaders.Metadata(data_path)
dataloader_train, dataloader_val, dataloader_test = loaders.get_dataloader(
    metadata, batch_size=BATCH_SIZE)

# get model
model = models.resnet101(num_classes=len(metadata.encoder.classes_),
                         model_save_path=model_save_path
                         ).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()

# train & fine tune
rounds = 0
for i in range(rounds):
    print('*'*100)
    print('*'*100)
    for param in model.parameters():
        param.requires_grad = False
    update_layers = [model.fc, model.layer4,
                     model.layer3, model.layer2, model.layer1]
    lrs = [1e-3, 1e-4, 1e-4, 1e-5, 1e-6]
    for layer, lr in zip(update_layers, lrs):
        model = train(model=model,
                      dataloaders=(dataloader_train, dataloader_val),
                      update_layer=layer,
                      train_on_val=False,
                      lr=lr,)
        # logger.info('fine tune on {}'.format(layer.__str__()))
        print('=' * 100)

# predict
# final train
logger.info('eval...')
loss_val, acc_val, labels_val = predict_eval(model, dataloader_val, loss_fn)
logger.info('loss_val:{}, acc_val:{}'.format(loss_val, acc_val))
logger.info('make submission')
__, __, labels_test = predict_eval(model, dataloader_test, loss_fn)
classes_test = metadata.encoder.inverse_transform(labels_test)
images_test = []
for image_name in metadata.data.images_test:
    images_test.append(os.path.basename(image_name))
submission = pd.DataFrame({'image': images_test,
                           'class': classes_test})
submission.to_csv(os.path.join(output_path, 'submission.csv'),
                  index=False, header=False)
__import__('ipdb').set_trace()
