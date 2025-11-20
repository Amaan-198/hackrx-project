"""
Manual script to download the HuggingFace embedding model.
Run this if you're having connection issues with the main app.
"""

import os
import sys

# Force online mode
os.environ["HF_HUB_OFFLINE"] = "0"

print("=" * 60)
print("HuggingFace Model Downloader")
print("=" * 60)
print()

print("Step 1: Testing internet connection to HuggingFace...")
try:
    import urllib.request
    response = urllib.request.urlopen('https://huggingface.co', timeout=10)
    print("✅ Successfully connected to HuggingFace.co")
    print(f"   Status Code: {response.status}")
except Exception as e:
    print(f"❌ Cannot connect to HuggingFace.co: {e}")
    print("\n⚠️  Possible solutions:")
    print("   1. Check your internet connection")
    print("   2. Check firewall/proxy settings")
    print("   3. Set proxy environment variables:")
    print("      set HTTP_PROXY=http://your-proxy:port")
    print("      set HTTPS_PROXY=http://your-proxy:port")
    sys.exit(1)

print()
print("Step 2: Downloading embedding model (all-MiniLM-L6-v2)...")
print("This may take a few minutes on first download...")
print()

try:
    # Try using huggingface_hub directly
    from huggingface_hub import snapshot_download
    import ssl
    
    # Disable SSL verification if needed (not recommended for production)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("Attempting to download model files...")
    cache_dir = snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    )
    print(f"✅ Model files downloaded to: {cache_dir}")
    
    # Now try loading with sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("✅ Model loaded successfully!")
    print()
    print("=" * 60)
    print("✅ SUCCESS! You can now run the Streamlit app.")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print()
    print("⚠️  Detailed error information:")
    import traceback
    traceback.print_exc()
    print()
    print("⚠️  Try these solutions:")
    print("   1. Ensure stable internet connection")
    print("   2. Check if you're behind a corporate firewall")
    print("   3. Try setting proxies:")
    print("      set HTTP_PROXY=http://your-proxy:port")
    print("      set HTTPS_PROXY=http://your-proxy:port")
    print("   4. Run with admin privileges if needed")
    print("   5. Check if antivirus is blocking downloads")
    sys.exit(1)
