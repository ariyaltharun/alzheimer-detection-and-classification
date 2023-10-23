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
