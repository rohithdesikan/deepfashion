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

# Import AWS packages
import sagemaker
import sagemaker_containers
import boto3
# from boto3.s3.connection import S3Connection

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Transform the incoming images by normalizing and re ordering and gather the labels and bounding boxes from the annotations
class TransformData(Dataset):
    def __init__(self, image_path, annos_path, filenames, resize=None):
        self.image_path = image_path
        self.annos_path = annos_path
        self.filenames = filenames

        if resize is not None:
            self.h = resize[0]
            self.w = resize[1]
    

    def __getitem__(self, index):

        # Get the image path and open the indexed image
        image_id = self.filenames[index] + '.jpg'
        path_image = os.path.join(self.image_path, image_id)
        img = Image.open(path_image)
        image_orig = np.array(img)

        # Move the numpy axes to the right order as accepted by Pytorch
        image_ordered = np.transpose(image_orig, (2, 0, 1))
        image_preprocessed = image_ordered/255.0
        if image.resize:
            tfrm = transforms.Resize((self.h, self.w))
            image_preprocessed = tfrm(image_preprocessed)

        image = torch.Tensor(image_preprocessed)

        # Load the annotations as a json
        annos_id = self.filenames[index] + '.json'
        path_anno = os.path.join(self.annos_path, annos_id)
        with open(path_anno,'r') as f:
            data = json.load(f)

        # Number of targets/labels/bounding boxes and initialize a final targets dictionary for PyTorch input
        num_targets = len([k for k in list(data.keys()) if 'item' in k])

        labels = [None] * num_targets
        boxes = [None] * num_targets
        # Iterate through each annotation to find the # of labels and associated bounding boxes in the image
        for j in range(1,num_targets+1):
            
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
        frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
        frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 14)
        for param in resnet.parameters():
            param.requires_grad_(True)

        modules = list(frcnn_model.children())
        self.frcnn = nn.Sequential(*modules)

    def forward(self, images):
        x = self.frcnn_model(images)
        return x


# Get the train data loader after performing the necessary preprocessing from the utils file
def _get_train_data_loader(bucket_name, prefix_name, train_batch_size, resize):
    logger.info("Get train data loader")

    batch_size = args.batch_size

    conn = S3Connection()
    bucket = conn.get_bucket(bucket_name)
    path_images = f"s3://{bucket_name}//{prefix_name}//{image}"
    path_annos = f"s3://{bucket_name}//{prefix_name}//{annos}"
    file_ids = []
    for file_id in images_path.list()[:100]:
        file_ids.append(file_id.split('.')[0])

    # Load the dataset, pass it through the preprocessing steps from the utils file and load the data loader
    dataset = TransformData(path_images, path_annos, file_ids, resize=resize)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, collate_fn = collate_fn)

    return data_loader

# Get the test data loader performing the same preprocessing as above
def _get_test_data_loader(bucket_name, prefix_name, test_batch_size, resize):
    logger.info("Get test data loader")
    conn = S3Connection()
    bucket = conn.get_bucket(bucket_name)
    path_images = f"s3://{bucket_name}//{prefix_name}//{image}"
    path_annos = f"s3://{bucket_name}//{prefix_name}//{annos}"
    file_ids = []
    for file_id in images_path.list()[100:]:
        file_ids.append(file_id.split('.')[0])

    # Load the dataset, pass it through the preprocessing steps from the utils file and load the data loader
    dataset = TransformData(path_images, path_annos, file_ids, resize=resize)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = test_batch_size, shuffle = True, collate_fn = collate_fn)

    return data_loader

# Save the model after training
def save_model(model, model_dir, fn):
    logger.info("Saving the latest model.")
    path = os.path.join(model_dir, f'{fn}.pt')
    torch.save(model.cpu().state_dict(), path)

# Load a saved model for testing
def load_model(model_dir, fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # with open(os.path.join(model_dir, f'{fn}.pth'), 'rb') as f:
    #     model.load_state_dict(torch.load(f))

    f = os.path.join(model_dir, f'{fn}.pth')
    model.load_state_dict(torch.load(f))
    return model.to(device)


def train(args):
    # use_cuda = args.num_gpus > 0
    # logger.debug("Number of gpus available - {}".format(args.num_gpus))
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(bucket_name, prefix_name, batch_size)
    test_loader = _get_test_data_loader(bucket_name, prefix_name, batch_size)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = FasterRCNN.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print("Epoch: ", epoch)
        model.train()
        for batch_idx, (batch_images, batch_annos) in enumerate(train_loader, 1):

            images, annos = list(batch_images.to(device)), list(batch_annos.to(device))

            optimizer.zero_grad()

            loss_dict = model(images, annos)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            optimizer.step()


            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), losses.item()))
        
    save_model(model, args.model_dir, args.fn)


def test(model, test_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, annos in test_loader:
            images, annos = images.to(device), annos.to(device)
            output = model(images)
            # test_loss += F.nll_loss(output, annos, size_average=False).item()  # sum up batch loss
    #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct += pred.eq(annos.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    # logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--resize', type=tuple, default=None, metavar='N',
                        help='Resize parameter. Default is (Height, Width) = (500, 400). Enter None for no resizing')
    parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
                        help='input batch size for testing (default: 30)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    # parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    # parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())