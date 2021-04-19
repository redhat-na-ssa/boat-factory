#!/usr/bin/env python3

import argparse
import collections
import os
import pathlib
import PIL.Image
import pprint
import random
import sys
import time
import uuid
import yaml


def parse_args():
    """ Read and parse the command line arugments. """
    parser = argparse.ArgumentParser(description="Generate some training data in YOLOv5 TXT format.")
    parser.add_argument('--asset-dir', '-a', type=str, nargs='?', default='./assets',
            help="Directory containing assets")
    parser.add_argument('--output-dir', '-o', type=str, nargs='?', default='./data',
            help="Directory to output training set to")
    parser.add_argument('--asset-max', type=int, nargs='?', default=10,
            help="Maximum possible assets per image")
    parser.add_argument('--asset-min', type=int, nargs='?', default=1,
            help="Minimum possible assets per image (0 will allow null images)")
    parser.add_argument('dataset', type=str, choices=('train', 'test', 'valid'), nargs=1,
            help="Type of data set to produce")
    parser.add_argument('count', metavar='COUNT', type=int, nargs=1,
            help="Number of sampels to create")
    args = parser.parse_args()
    return {"assets": args.asset_dir,
            "output": args.output_dir,
            "dataset": args.dataset[0],
            "count": args.count[0],
            "amin": args.asset_min,
            "amax": args.asset_max}


class ImageObjectGenerator(object):
    """ Generator class for creating PIL Images on demand with cached image path. """

    def __init__(self, path):
        # cache file path
        self._path = os.path.abspath(path)

        # ensure file can be found and opened
        image = self()

        # cache image attributes
        self._size = image.size

        # clean up
        image.close()

    def __call__(self, *args, **kwargs):
        # Generate an instance on Pil.Image.Image
        return PIL.Image.open(self._path, *args, **kwargs)

    def __repr__(self):
        # String representation of self
        return 'ImageObjectGenerator("{0}")'.format(self._path)

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return os.path.basename(self._path)


def image_crawl(basepath, relpath):
    """ Crawl directory for images. """
    cat_dir = os.path.abspath(os.path.join(basepath, relpath))
    (_, _, file_names) = next(os.walk(cat_dir))
    output = []

    for file_name in file_names:
        # Ensure file can be read as an image
        try:
            new_image = ImageObjectGenerator(os.path.join(cat_dir, file_name))
        except OSError:
            continue
        output.append(new_image)

    return output


def load_assets(library):
    """ Load all of the assets from the asset library. """
    with open(os.path.join(library, 'data.yaml')) as toc_file:
        toc = yaml.load(toc_file, Loader=yaml.SafeLoader)

    assets = {'categories': {}, 'backdrops': []}

    # read all specified categories and their sprites
    for (name, relpath) in toc.get('categories', {}).items():
        # search for all images in each category
        assets['categories'][name] = image_crawl(library, relpath)


    # read all of the backdrops
    relpath = toc.get('backdrops')
    if relpath:
        assets['backdrops'] = image_crawl(library, relpath)

    # create metadata
    assets["_meta"] = {"nbackdrops": len(assets['backdrops']),
                       "ncats": len(assets['categories'].keys()),
                       "ncat_assets": {key: len(val) for key, val in assets['categories'].items()}}

    return assets


def init_output_dir(output_dir, dataset, assets):
    """ Initialize the output directory structure. """
    # create directory structure
    full_output_dir = os.path.abspath(os.path.join(output_dir, dataset))
    images_out = os.path.join(full_output_dir, 'images')
    label_out = os.path.join(full_output_dir, 'labels')
    pathlib.Path(images_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(label_out).mkdir(parents=True, exist_ok=True)
    output = {"images": images_out, "labels": label_out}

    # read existing toc
    try:
        with open(os.path.join(output_dir, 'data.yaml')) as toc_file:
            toc = yaml.load(toc_file, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        toc = {}
    # map dataset name to name expected in toc
    toc_keys = {"valid": "val", "train": "train", "test": None}
    toc_dataset = toc_keys.get(dataset)
    # update names database
    toc_names = toc.get('names', [])
    new_names = assets['categories']
    all_names = collections.OrderedDict(zip(toc_names, [0]*len(toc_names)))
    all_names.update(new_names)
    names = list(all_names.keys())
    output.update({"names": names, "nc": len(names)})

    # write update toc
    if toc_dataset:
        toc[toc_dataset] = '../{0}/images'.format(dataset)
    toc['nc'] = output['nc']
    toc['names'] = output['names']
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as toc_file:
        yaml.dump(toc, toc_file, default_flow_style=False)

    return output


class BlockTimer(object):
    def __init__(self, name):
        self._name = name
        self._start = time.time()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end()

    def start(self):
        self._start = time.time()
        sys.stdout.write('\n{0}\n\n'.format(self._name))

    def end(self):
        elap = time.time() - self._start
        sys.stdout.write('\n{0} completed in {1} seconds.\n'.format(self._name, elap))
        print('')


def create_image(amin, amax, assets, output):
    """ Create a synthetic image using the provided asset library. """
    # choose backdrop
    image_gen = random.choice(assets['backdrops'])
    image = image_gen()

    # place assets
    boxes = []
    totals = {}
    for asset_count in range(random.randint(amin, amax)):
        # pick category
        cat = random.choice(list(assets['categories'].keys()))
        cat_id = output['names'].index(cat)
        totals[cat] = totals.get(cat, 0) + 1

        # pick asset
        asset = random.choice(assets['categories'][cat])()

        # rotate the asset
        asset = asset.rotate(random.randint(0, 360), PIL.Image.NEAREST, expand=1)

        # place the asset
        top_left = (random.randint(0 - asset.size[0], image.size[0]),
                    random.randint(0 - asset.size[1], image.size[1]))
        image.paste(asset, top_left, asset)
        asset.close()

        # save the box
        boxes +=[
            (cat_id,  # category id
             max(0, (asset.size[0] / 2 + top_left[0]) / image.size[0]),  # normalized x center
             max(0, (asset.size[1] / 2 + top_left[1]) / image.size[1]),  # normalized y center
             (min(image.size[0], top_left[0] + asset.size[0]) - max(0, top_left[0])) / image.size[0],  # norm width
             (min(image.size[1], top_left[1] + asset.size[1]) - max(0, top_left[1])) / image.size[1]  # norm height
            )]

    return (image, boxes, {"backdrop": image_gen.name, "totals" : totals})


def save_image(fname, output, image):
    """ Save the image that has been generated in the proper dataset. """
    full_fname_png = os.path.join(output['images'], fname) + '.png'
    full_fname_txt = os.path.join(output['labels'], fname) + '.txt'

    image[0].save(full_fname_png, 'PNG')
    with open(full_fname_txt, 'w') as txt:
        for row in image[1]:
            txt.write('{0} {1} {2} {3} {4}\n'.format(*row))


if __name__ == "__main__":
    # parse user inputs
    options = parse_args()

    with BlockTimer('Loading data'):
        # load data
        assets = load_assets(options['assets'])
        output = init_output_dir(options['output'], options['dataset'], assets)

        # print parsed inputs
        pp = pprint.PrettyPrinter(indent=2)
        print('Assets Library:')
        pp.pprint(assets)
        print('')
        print('Output Information:')
        pp.pprint(output)

        # health checks
        if len(assets['backdrops']) == 0:
            raise LookupError("Could not find any backdrops.")
        if sum([len(lib) for lib in assets['categories'].values()]) == 0:
            raise LookupError("Could not find any assets.")

    with BlockTimer('Generating images'):
        # generate images
        for count in range(1, options['count']+1):
            sys.stdout.write('[{0} / {1}]  '.format(count, options['count']))
            fname = uuid.uuid4().hex
            synthetic_image = create_image(options['amin'], options['amax'], assets, output)
            sys.stdout.write('{0}:\n    '.format(fname))
            save_image(fname, output, synthetic_image)
            pp.pprint(synthetic_image[2])
