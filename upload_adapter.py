import boto3
import os
import argparse
import datetime
import sys

def upload_directory(path, bucket_name, s3_prefix):
    s3 = boto3.client('s3')
    
    if not os.path.exists(path):
        print(f"Error: Directory {path} does not exist.")
        return False

    print(f"Uploading {path} to s3://{bucket_name}/{s3_prefix}")

    files_uploaded = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, path)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/") # Ensure forward slashes

            print(f"  - Uploading {file}...")
            try:
                s3.upload_file(local_path, bucket_name, s3_path)
                files_uploaded += 1
            except Exception as e:
                print(f"Failed to upload {file}: {e}")
                return False

    if files_uploaded == 0:
        print("Warning: No files were uploaded.")
    else:
        print(f"Successfully uploaded {files_uploaded} files.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to S3")
    parser.add_argument("--character", required=True, help="Character name")
    parser.add_argument("--path", default="./output_adapters", help="Path to adapter files")
    parser.add_argument("--bucket", required=False, help="S3 Bucket Name (overrides env var S3_BUCKET)")
    
    args = parser.parse_args()

    # Get bucket from arg or env
    bucket_name = args.bucket or os.environ.get("S3_BUCKET")
    
    if not bucket_name:
        # Try to guess or fail
        print("Error: S3 Bucket not specified. Set S3_BUCKET env var or pass --bucket.")
        # Attempt to list buckets and find one with 'uploads' in the name for convenience? 
        # No, that's risky.
        sys.exit(1)

    # Create timestamp prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Prefix: loras/{character_name}/{timestamp}/
    # Sanitize character name
    char_safe = args.character.replace(" ", "_").replace(".", "_")
    prefix = f"loras/{char_safe}/{timestamp}"
    
    success = upload_directory(args.path, bucket_name, prefix)
    
    if not success:
        sys.exit(1)
