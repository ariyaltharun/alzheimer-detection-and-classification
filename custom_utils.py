def create_csv(root: Path, pd: pd = pd):
    """Creates csv dataset from the given images
    Parameters:
    - root -> Root path of dataset folder

    Output: 
    - Creates two file `train.csv` and `test.csv`
    """

    for status in root.iterdir():
        train = {
            'img_path': [],
            'labels': []
        }
        test = {
            'img_path': [],
            'labels': []
        }
        class_names = root / status
        for class_name in class_names.iterdir():
            images = root / class_name
            img_path = f"Dataset/{class_name}/"
            for image in images.iterdir():
                if status == 'train':
                    train['img_path'].append(img_path+str(image))
                    train['labels'].append(class_name)
                if status == 'test':
                    test['img_path'].append(img_path+str(image))
                    test['labels'].append(class_name)

    pd.DataFrame(train).to_csv("train.csv", index=False)
    pd.DataFrame(test).to_csv("test.csv", index=False)
