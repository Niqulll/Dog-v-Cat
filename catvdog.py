import fastbook
import fastai
fastbook.setup_book()
from duckduckgo_search import *
from fastbook import *
from fastai.vision.all import *
from fastdownload import download_url
from time import sleep

def search_images(term, max_images=50):
    print(f"Searching for '{term}'")
    return (L(search_images_ddg(term, max_images=max_images)))

if __name__ == '__main__':
    searches = 'cat', 'dog'
    path = Path('images/cat_or_dog')

    #for o in searches:
    #    dest = (path/o)
    #    dest.mkdir(exist_ok=True, parents=True)
    #    download_images(dest, urls=search_images(f'single {o} photo'))
    #    sleep(10)
    #    resize_images(path/o, max_size=400, dest=path/o)

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)


    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    cat_or_dog,_,probs = learn.predict(PILImage.create('images/dog.jpg'))
    print(f"This is a: {cat_or_dog}.")
    print(f"Probability: {probs[0]:.4f}")
