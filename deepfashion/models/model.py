# Import base packages
import numpy as np
import argparse
import json
import logging
import os
import sys
from PIL import Image

# Import PyTorch packages
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Transform the incoming images by normalizing and re ordering and gather the labels and bounding boxes from the annotations
class TransformData(Dataset):
    def __init__(self, images_path, targets_path, filenames, resize=None):
        self.images_path = images_path
        self.targets_path = targets_path
        self.filenames = filenames
        self.resize = True if resize is not None else False

        if self.resize:
            self.h = resize[0]
            self.w = resize[1]
    

    def __getitem__(self, index):

        # Get the image path and open the indexed image
        image_id = self.filenames[index] + '.jpg'
        path_image = os.path.join(self.images_path, image_id)
        img = Image.open(path_image)
        image_orig = np.array(img)

        # Move the numpy axes to the right order as accepted by Pytorch
        image_ordered = np.transpose(image_orig, (2, 0, 1))
        image_preprocessed = image_ordered/255.0
        if self.resize:
            tfrm = transforms.Resize((self.h, self.w))
            image_preprocessed = tfrm(image_preprocessed)

        image = torch.Tensor(image_preprocessed)

        # Load the annotations as a json
        target_id = self.filenames[index] + '.json'
        path_target = os.path.join(self.targets_path, target_id)
        with open(path_target,'r') as f:
            data = json.load(f)

        # Number of targets/labels/bounding boxes and initialize a final targets dictionary for PyTorch input
        num_targets = len([k for k in list(data.keys()) if 'item' in k])

        labels = [None] * num_targets
        boxes = [None] * num_targets

        # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
        for j in range(1, num_targets+1):
            
            # Obtain the image label
            label = data[f'item{j}']['category_id']

            # Get the bounding box and normalize it by the width and height calculated above
            x0, y0, x1, y1 = data[f'item{j}']['bounding_box']

            # Put the bounding box information by label into a specific list
            box = [x0, y0, x1, y1]

            # Convert the list of labels into a tensor
            labels[j-1] = label

            # Convert the list of lists of bounding boxes to a torch float tensor
            boxes[j-1] = box


        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.Tensor(boxes)
    
        targets = {'boxes': boxes, 'labels': labels}

        return image, targets
    
    # Return the length of these filenames
    def __len__(self):
        return len(self.filenames)

# How to collate the files within each batch. This gets loaded into the train and test loaders
def collate_fn(batch):
    return list(zip(*batch))


# Set up the Pytorch model in a FasterRCNN class and set all parameters to be trainable
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.frcnn_model.roi_heads.box_predictor.cls_score.in_features
        self.frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)
        # params = [p for p in self.frcnn_model.parameters() if p.requires_grad]

        # modules = list(frcnn_model.children())
        # self.frcnn = nn.Sequential(*modules)

    def forward(self, images, targets):
        x = self.frcnn_model(images, targets)

        return x


# Get the train data loader after performing the necessary preprocessing from the utils file
def _get_train_data_loader(resize):
    logger.info("Get train data loader")

    images_path = os.path.join(args.train_dir, 'image')
    targets_path = os.path.join(args.train_dir, 'annos')
    file_ids = [file_id.split('.')[0] for file_id in os.listdir(images_path)]

    # Load the dataset, pass it through the preprocessing steps from the utils file and load the data loader
    dataset = TransformData(images_path, targets_path, file_ids, resize=resize)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn)

    return train_loader

# Get the test data loader performing the same preprocessing as above
def _get_test_data_loader(resize):
    logger.info("Get test data loader")

    images_path = os.path.join(args.test_dir, 'image')
    targets_path = os.path.join(args.test_dir, 'annos')
    file_ids = [file_id.split('.')[0] for file_id in os.listdir(images_path)]

    # Load the dataset, pass it through the preprocessing steps from the utils file and load the data loader
    dataset = TransformData(images_path, targets_path, file_ids, resize=resize)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)

    return test_loader

# Save the model after training
def save_model(model, fn):
    logger.info("Saving the latest model.")
    path = os.path.join(args.model_directory, f'{fn}.pt')
    torch.save(model.cpu().state_dict(), path)

# Load a saved model for testing
def load_model(fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # with open(os.path.join(model_dir, f'{fn}.pth'), 'rb') as f:
    #     model.load_state_dict(torch.load(f))

    f = os.path.join(args.model_directory, f'{fn}.pt')
    model = torch.load_state_dict(torch.load(f))
    return model.to(device)


def train(args):
    # use_cuda = args.num_gpus > 0
    # logger.debug("Number of gpus available - {}".format(args.num_gpus))
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.resize)
    test_loader = _get_test_data_loader(args.resize)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    model = FasterRCNN().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print("Epoch: ", epoch)

        model.train()

        for batch_idx, (batch_images, batch_targets) in enumerate(train_loader):

            images = list(img.to(device) for img in batch_images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            model.zero_grad()
            losses.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), losses.item()))
        
    save_model(model, args.model_directory, args.fn)


# def test(model, test_loader):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.eval()
#     test_loss = 0
#     correct = 0

#     with torch.no_grad():
#         for batch_idx, (batch_images, batch_targets) in enumerate(test_loader):
#             images = list(img.to(device) for img in batch_images)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
#             output = model(images)

            # test_loss += F.nll_loss(output, annos, size_average=False).item()  # sum up batch loss
    #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct += pred.eq(annos.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    # logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    # return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                        help='input batch size for training and testing (default: 10)')
    parser.add_argument('--resize', type=tuple, default=None, metavar='N',
                        help='Resize parameter. Default is (Height, Width) = (500, 400). Enter None for no resizing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fn', type=str, default='20200712_model_V1',
                        help='model name for saved model')
    parser.add_argument('--bucket-name', type=str, default='s3://rohithdesikan-deepfashion/deepfashion-sample',
                        help='This is where the data is currently stored')
    parser.add_argument('--model-directory', type=str, default='s3://rohithdesikan-deepfashion/deepfashion-sample/output',
                        help='This is the where the output of the model should be stored')
    parser.add_argument('--path-images', type=str, default='s3://rohithdesikan-deepfashion/deepfashion-sample/image',
                        help='Images path')
    parser.add_argument('--path-targets', type=str, default='s3://rohithdesikan-deepfashion/deepfashion-sample/annos',
                        help='Image annotations path')

    # Container environment
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--module_dir', type=str, default='s3://rohithdesikan-deepfashion/output')
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))


    args = parser.parse_args()

    train(args)