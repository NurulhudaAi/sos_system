#!/usr/bin/env python3
"""
test_mongodb.py — ทดสอบการเชื่อม MongoDB
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pymongo

# Load .env
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("MONGODB_DB_NAME", "iam")

if not MONGODB_URI:
    print("❌ MONGODB_URI not set in .env")
    sys.exit(1)

print("\n" + "="*60)
print("🔗 MongoDB Connection Tester")
print("="*60)
print(f"URI:    {MONGODB_URI[:50]}...")
print(f"DB:     {DB_NAME}")
print("="*60 + "\n")

try:
    print("⏳ Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

    # Test connection
    print("⏳ Testing server connection...")
    client.admin.command('ping')
    print("✅ Server ping successful")

    # Get database
    print(f"⏳ Accessing database: {DB_NAME}...")
    db = client[DB_NAME]
    print("✅ Database accessible")

    # List collections
    collections = db.list_collection_names()
    print(f"✅ Collections found: {len(collections)}")
    for col in collections:
        count = db[col].count_documents({})
        print(f"   • {col}: {count} docs")

    # Try a simple insert (optional)
    print("\n⏳ Testing write permission...")
    test_col = db["_connection_test"]
    result = test_col.insert_one({"test": "connection", "timestamp": __import__("datetime").datetime.now()})
    print(f"✅ Write successful: {result.inserted_id}")
    test_col.delete_one({"_id": result.inserted_id})
    print("✅ Cleanup done")

    print("\n" + "="*60)
    print("✅ All tests passed! Connection is working.")
    print("="*60 + "\n")

except pymongo.errors.ServerSelectionTimeoutError:
    print("\n❌ Connection timeout!")
    print("   • Check internet connection")
    print("   • Check MONGODB_URI format")
    print("   • Check MongoDB Atlas IP Whitelist")
    sys.exit(1)

except pymongo.errors.OperationFailure as e:
    print(f"\n❌ Authentication failed: {e}")
    print("   • Check username/password")
    print("   • Check credentials haven't expired")
    print("   • Try resetting password in MongoDB Atlas")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    if 'client' in locals():
        client.close()
