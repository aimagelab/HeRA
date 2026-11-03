#!/usr/bin/env python3
"""
Pack LLaVA images into LMDB files to reduce filesystem usage.

This script reads a JSON annotation file, packs images into multiple LMDB databases,
and creates a new annotation file with LMDB references while maintaining the exact
order of the input JSON.
"""

import json
import lmdb
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Pack LLaVA images into LMDB files for efficient storage'
    )
    parser.add_argument(
        '--annotations',
        required=True,
        help='Path to input JSON annotations file'
    )
    parser.add_argument(
        '--image_root',
        required=True,
        help='Root directory containing the images'
    )
    parser.add_argument(
        '--lmdb_root',
        required=True,
        help='Root directory where LMDB files will be created'
    )
    parser.add_argument(
        '--output_json',
        required=True,
        help='Path to output JSON with LMDB references'
    )
    parser.add_argument(
        '--images_per_db',
        type=int,
        default=5000,
        help='Number of images per LMDB file (default: 5000)'
    )
    parser.add_argument(
        '--db_prefix',
        type=str,
        default='llava_imgs',
        help='Prefix for LMDB database names (default: llava_imgs)'
    )

    args = parser.parse_args()

    print(f"Loading annotations from {args.annotations}")
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)

    print(f"Total samples: {len(annotations)}")

    # Create lmdb_root if it doesn't exist
    os.makedirs(args.lmdb_root, exist_ok=True)

    # First pass: collect all samples with images and assign them to LMDB files
    samples_with_images = []
    for idx, sample in enumerate(annotations):
        if 'image' in sample and sample['image']:
            samples_with_images.append((idx, sample))

    print(f"Samples with images: {len(samples_with_images)}")
    print(f"Text-only samples: {len(annotations) - len(samples_with_images)}")

    # Calculate number of LMDB files needed
    num_dbs = (len(samples_with_images) + args.images_per_db - 1) // args.images_per_db
    print(f"\nCreating {num_dbs} LMDB database(s)...")

    # Create a mapping from original index to LMDB file
    idx_to_lmdb = {}

    # Process images in batches and create LMDB files
    for db_idx in range(num_dbs):
        batch_start = db_idx * args.images_per_db
        batch_end = min((db_idx + 1) * args.images_per_db, len(samples_with_images))
        batch_samples = samples_with_images[batch_start:batch_end]

        # Create LMDB directory name
        lmdb_name = f'{args.db_prefix}_{db_idx:05d}'
        lmdb_path = os.path.join(args.lmdb_root, lmdb_name)

        print(f"\n[{db_idx+1}/{num_dbs}] Creating {lmdb_name} with {len(batch_samples)} images")

        # Create LMDB environment
        env = lmdb.open(
            lmdb_path,
            map_size=1099511627776,  # 1TB
            writemap=True,
            map_async=True,
            metasync=False,
            sync=False
        )

        successful_writes = 0
        failed_writes = 0

        with env.begin(write=True) as txn:
            for orig_idx, sample in tqdm(batch_samples, desc=f'  Packing images'):
                image_rel_path = sample['image']
                image_full_path = os.path.join(args.image_root, image_rel_path)

                try:
                    # Read image as bytes
                    with open(image_full_path, 'rb') as img_file:
                        img_data = img_file.read()

                    # Store in LMDB with the relative image path as key
                    txn.put(image_rel_path.encode('utf-8'), img_data)

                    # Map original index to LMDB file
                    idx_to_lmdb[orig_idx] = lmdb_name
                    successful_writes += 1

                except FileNotFoundError:
                    print(f"\n  Warning: Image not found: {image_full_path}")
                    failed_writes += 1
                except Exception as e:
                    print(f"\n  Error processing {image_full_path}: {e}")
                    failed_writes += 1

        # Sync and close the environment
        env.sync()
        env.close()

        print(f"  Successfully packed: {successful_writes}")
        if failed_writes > 0:
            print(f"  Failed: {failed_writes}")

    # Second pass: create output JSON in the same order as input
    print(f"\nCreating output JSON with same order as input...")
    output_annotations = []

    for idx, sample in enumerate(annotations):
        output_sample = sample.copy()

        # Add lmdb_file field if this sample has an image that was packed
        if idx in idx_to_lmdb:
            output_sample['lmdb_file'] = idx_to_lmdb[idx]

        output_annotations.append(output_sample)

    # Save output JSON
    print(f"Saving annotations to {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(output_annotations, f, indent=2)

    print("\nDone!")
    print(f"  Created {num_dbs} LMDB database(s) in {args.lmdb_root}")
    print(f"  Total samples in output: {len(output_annotations)}")
    print(f"  Order preserved: ✓")


if __name__ == '__main__':
    main()
