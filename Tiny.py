import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
# CLASS_LIST_FILE = '/apdcephfs/share_1290939/jiaxiaojun/imagenet/tiny-imagenet-200/wnids.txt'
# VAL_ANNOTATION_FILE = '/apdcephfs/share_1290939/jiaxiaojun/imagenet/tiny-imagenet-200/val/val_annotations.txt'



class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root,class_per_num=500, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.transform_base=transforms.Compose([
                            transforms.ToTensor(),])
        self.CLASS_LIST_FILE='wnids.txt'
        self.VAL_ANNOTATION_FILE='val_annotations.txt'
        self.NUM_IMAGES_PER_CLASS=class_per_num
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        self.images1 = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            if self.split=='train':
                self.images = [self.read_image(path,self.transform[0]) for path in self.image_paths]
                self.images1 = [self.read_image(path, self.transform[1]) for path in self.image_paths]
                for i in range(len(self.images)):
                    if self.images[i].size()[0]==1:
                        print(self.image_paths[i])
            else:
                self.images = [self.read_image(path, self.transform[0]) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            if self.split=='train':
                img1=self.images[index]
                img2=self.images1[index]
            else:
                img = self.images[index]

        else:
            if self.split != 'train':
                img = self.read_image(file_path,self.transform[0])
            else:
                img1=self.read_image(file_path,self.transform[0])
                img2 = self.read_image(file_path,self.transform[1])



        if self.split == 'train':
            return img1,img2, self.labels[os.path.basename(file_path)]
        else:
            if self.split == 'test':
                return img
            else:
                # file_name = file_path.split('/')[-1]
                return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path,transform=None):
        img = Image.open(path)
        return transform(img) if transform else img