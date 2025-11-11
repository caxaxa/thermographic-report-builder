#!/usr/bin/env python3
"""
LaTeX Compiler for Solar Report PDFs
Downloads LaTeX bundle from S3, compiles to PDF, uploads result back to S3.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


def log(message: str):
    """Print log message with timestamp."""
    print(f"[LaTeX Compiler] {message}", flush=True)


def download_s3_directory(s3_client, bucket: str, prefix: str, local_dir: Path):
    """Download entire S3 directory to local path."""
    log(f"Downloading s3://{bucket}/{prefix} to {local_dir}")

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    file_count = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Get relative path
            rel_path = key[len(prefix):].lstrip('/')
            if not rel_path:  # Skip the prefix itself
                continue

            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            log(f"  Downloading {key}")
            s3_client.download_file(bucket, key, str(local_file))
            file_count += 1

    log(f"Downloaded {file_count} files")
    return file_count


def compile_latex(tex_file: Path, work_dir: Path) -> tuple[bool, str]:
    """
    Compile LaTeX file to PDF.

    Returns:
        (success, error_message or pdf_path)
    """
    log(f"Compiling {tex_file}")

    # Run pdflatex twice for proper cross-references
    for run in [1, 2]:
        log(f"pdflatex run {run}/2")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_file.name)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            errors='replace'  # Handle non-UTF-8 characters from pdflatex output
        )

        if result.returncode != 0:
            error_msg = f"pdflatex failed on run {run}:\n{result.stdout}\n{result.stderr}"
            log(f"ERROR: {error_msg}")

            # Save full log
            log_file = work_dir / "latex_error.log"
            with open(log_file, "w") as f:
                f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")

            return False, error_msg

    pdf_file = tex_file.with_suffix('.pdf')
    if pdf_file.exists():
        size_mb = pdf_file.stat().st_size / 1_000_000
        log(f"PDF generated successfully: {size_mb:.1f} MB")
        return True, str(pdf_file)
    else:
        return False, "PDF file not generated (unknown error)"


def compress_pdf(input_pdf: Path, output_pdf: Path) -> bool:
    """Compress PDF using ghostscript."""
    log(f"Compressing PDF: {input_pdf.name}")

    result = subprocess.run([
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",  # Good balance of quality/size
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
        str(input_pdf)
    ], capture_output=True)

    if result.returncode == 0 and output_pdf.exists():
        original_mb = input_pdf.stat().st_size / 1_000_000
        compressed_mb = output_pdf.stat().st_size / 1_000_000
        ratio = (1 - compressed_mb / original_mb) * 100
        log(f"Compressed: {original_mb:.1f} MB -> {compressed_mb:.1f} MB ({ratio:.0f}% reduction)")
        return True
    else:
        log(f"Compression failed: {result.stderr.decode()}")
        return False


def upload_to_s3(s3_client, local_file: Path, bucket: str, key: str):
    """Upload file to S3."""
    size_mb = local_file.stat().st_size / 1_000_000
    log(f"Uploading {local_file.name} ({size_mb:.1f} MB) to s3://{bucket}/{key}")
    s3_client.upload_file(str(local_file), bucket, key)
    log(f"Upload complete: s3://{bucket}/{key}")


def main():
    """Main entrypoint."""
    log("=" * 80)
    log("Starting LaTeX PDF Compilation")
    log("=" * 80)

    # Get environment variables
    project_id = os.environ.get('SOLAR_PROJECT_ID')
    user_id = os.environ.get('SOLAR_USER_ID')
    aws_region = os.environ.get('SOLAR_AWS_REGION', 'us-east-2')
    reports_bucket = os.environ.get('SOLAR_REPORTS_BUCKET', 'solar-reports-prod')

    if not project_id or not user_id:
        log("ERROR: Missing required environment variables SOLAR_PROJECT_ID or SOLAR_USER_ID")
        return 1

    log(f"Project: {project_id}, User: {user_id}")
    log(f"Reports bucket: {reports_bucket}")

    # Setup
    work_dir = Path("/work")
    s3_client = boto3.client('s3', region_name=aws_region)

    try:
        # Step 1: Download LaTeX bundle
        log("=" * 80)
        log("STEP 1: Downloading LaTeX bundle from S3")
        log("=" * 80)

        tex_bundle_prefix = f"{user_id}/projects/{project_id}/tex_bundle"
        download_s3_directory(s3_client, reports_bucket, tex_bundle_prefix, work_dir)

        # Step 2: Compile LaTeX
        log("=" * 80)
        log("STEP 2: Compiling LaTeX to PDF")
        log("=" * 80)

        tex_file = work_dir / "report.tex"
        if not tex_file.exists():
            log(f"ERROR: {tex_file} not found!")
            return 1

        success, result = compile_latex(tex_file, work_dir)
        if not success:
            log(f"Compilation failed: {result}")
            # Upload error log if it exists
            error_log = work_dir / "latex_error.log"
            if error_log.exists():
                upload_to_s3(
                    s3_client,
                    error_log,
                    reports_bucket,
                    f"{user_id}/projects/{project_id}/thermographic-report/latex_error.log"
                )
            return 1

        pdf_full = Path(result)

        # Step 3: Compress PDF
        log("=" * 80)
        log("STEP 3: Compressing PDF")
        log("=" * 80)

        pdf_compressed = work_dir / "report-compressed.pdf"
        compress_success = compress_pdf(pdf_full, pdf_compressed)

        # Step 4: Upload results
        log("=" * 80)
        log("STEP 4: Uploading PDFs to S3")
        log("=" * 80)

        # Upload full resolution PDF
        upload_to_s3(
            s3_client,
            pdf_full,
            reports_bucket,
            f"{user_id}/projects/{project_id}/thermographic-report/report-full.pdf"
        )

        # Upload compressed PDF
        if compress_success:
            upload_to_s3(
                s3_client,
                pdf_compressed,
                reports_bucket,
                f"{user_id}/projects/{project_id}/thermographic-report/report-lowres.pdf"
            )
        else:
            # If compression failed, use full PDF as lowres too
            upload_to_s3(
                s3_client,
                pdf_full,
                reports_bucket,
                f"{user_id}/projects/{project_id}/thermographic-report/report-lowres.pdf"
            )

        log("=" * 80)
        log("✅ PDF compilation completed successfully")
        log("=" * 80)
        return 0

    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
