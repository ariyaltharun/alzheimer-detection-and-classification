def create_csv(root: Path, pd: pd = pd):
    """Creates csv dataset from the given images
    Parameters:
    - root -> Root path of dataset folder

    Output: 
    - Creates two file `train.csv` and `test.csv`
    """
    train = {
        'img_path': [],
        'labels': []
    }
    test = {
        'img_path': [],
        'labels': []
    }
    for status in root.iterdir():
        class_names = root / status
        for class_name in class_names.iterdir():
            images = root / class_name
            for image in images.iterdir():
                if status == 'train':
                    train['img_path'].append("Dataset/"+str(image))
                    train['labels'].append(str(class_name)[6:])
                if status == 'test':
                    test['img_path'].append("Dataset/"+str(image))
                    test['labels'].append(str(class_name)[5:])

    pd.DataFrame(train).to_csv("train.csv", index=False)
    pd.DataFrame(test).to_csv("test.csv", index=False)


class AlzheimerDataset(Dataset):
    """ Alzheimer Dataset that contains 4 classes """
    def __init__(self, csv_file, root_dir=None, transform=None, target_transform):
        """
        Arguments:
        ----------
        csv_file: dataset file in csv format
        root_dir: root directory to dataset folder if any [DEFAULT: None]
        transform: any transform if required [DEFAULT: None]
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    
    def __len__(self):
        """
        Returns:
        -------
        Total number of instances in dataset
        """
        return len(self.csv_file)

    def __getitem__(self, idx):
        """
        Arguments:
        ----------
        idx: index to select the particular instance

        Output:
        -------
        tuple (img, label)
        img: torch image shape (1, 3, 64, 64) (format: [N, C, W, H]) 
        label: label associated with image
        """
        img_path = self.csv_file.iloc[idx, 0]
        # TODO: img = read_img(img_path)
        label = self.csv_file.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
        return img_path, self.classes.index(label)
        