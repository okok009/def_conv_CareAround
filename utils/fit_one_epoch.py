import datetime
import torch
import os
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, save_period=10, save_dir='checkpoints'):
    train_data_loader.reset()
    print('---------------start training---------------')
    model.train()
    with tqdm(total=train_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        for i in range(train_iter):
            train_batch_loss = 0
            img, bboxes, labels = train_data_loader()
            targets = []
            for j in range(img.shape[0]):
                d = {}
                d['boxes'] = bboxes[j]
                d['labels'] = labels[j]
                targets.append(d)
            loss_dict = model(img, targets)
            train_batch_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'batch_loss'    : train_batch_loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
    lr_scheduler.step()
    # print('---------------start validate---------------')
    # model.eval()
    # val_data_loader.reset()
    # with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
    #     val_batch_loss = 0
    #     with torch.no_grad():
    #         for i in range(val_iter):
    #             img, bboxes, labels = val_data_loader()
    #             pred = model(img)
    #             val_batch_loss =??

    #             pbar.set_postfix(**{'val_loss'    : val_batch_loss/(i+1)})
    #             pbar.update(1)

    # print(f'\ntrain loss:{train_batch_loss} || val loss:{val_batch_loss/(val_iter)}\n')

    if epoch % save_period == 0 or epoch == epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, f'ep{epoch}-loss{train_batch_loss}.pth'))
        # torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch, train_batch_loss, val_batch_loss/(val_iter))))
