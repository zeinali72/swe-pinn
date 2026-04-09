#!/usr/bin/env python3
"""One-time cleanup: delete empty W&B artifact collections.

Usage:
    python scripts/cleanup_wandb_artifacts.py [--dry-run]
"""
import argparse

import wandb


def main():
    parser = argparse.ArgumentParser(description="Delete empty W&B artifact collections")
    parser.add_argument("--dry-run", action="store_true", help="List but don't delete")
    parser.add_argument("--entity", default="zeinali72-exeter")
    parser.add_argument("--project", default="swe-pinn")
    args = parser.parse_args()

    api = wandb.Api()
    collections = api.artifact_type_collections(args.project, args.entity)

    deleted = 0
    for col in collections:
        versions = list(col.versions())
        if len(versions) == 0:
            deleted += 1
            if args.dry_run:
                print(f"[dry-run] Would delete: {col.name} (type={col.type})")
            else:
                try:
                    col.delete()
                    print(f"Deleted: {col.name} (type={col.type})")
                except Exception as e:
                    print(f"Failed to delete {col.name}: {e}")

    print(f"\n{'Would delete' if args.dry_run else 'Deleted'} {deleted} empty collections.")


if __name__ == "__main__":
    main()
