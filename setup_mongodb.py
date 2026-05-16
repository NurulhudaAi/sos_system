#!/usr/bin/env python3
"""MongoDB setup and connection test for CCTV SOS System."""

import sys
import os
from pathlib import Path

# Ensure env is loaded
from config.env_manager import init_env
init_env(require_edit=False)

def test_connection():
    """Test MongoDB connection and create collections/indexes."""
    import database
    from pymongo import MongoClient

    print("\n" + "="*60)
    print("MongoDB Connection & Setup Test")
    print("="*60 + "\n")

    # Get connection details
    uri = os.getenv("MONGODB_URI", "")
    db_name = os.getenv("MONGODB_DB_NAME", "sos_system")

    if not uri:
        print("❌ MONGODB_URI not set in .env")
        return False

    print(f"Testing connection to: {uri[:50]}...")
    print(f"Database name: {db_name}")

    try:
        # Test connection
        client = MongoClient(
            uri,
            retryWrites=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
        )

        # Ping to test auth
        print("\n⏳ Testing authentication...")
        client.admin.command("ping")
        print("✅ Connection successful!")

        # Get database
        db = client[db_name]

        # List existing collections
        print(f"\n📊 Collections in database '{db_name}':")
        collections = db.list_collection_names()
        if collections:
            for col in collections:
                count = db[col].count_documents({})
                print(f"   ✓ {col:20} ({count} documents)")
        else:
            print("   (empty - will create on first insert)")

        # Create indexes
        print("\n⏳ Creating indexes...")
        database._ensure_indexes()
        print("✅ Indexes created/verified")

        # Test insert
        print("\n⏳ Testing insert operation...")
        import uuid
        test_event = {
            "event_uuid": str(uuid.uuid4()),
            "event_type": "test",
            "created_at": __import__('datetime').datetime.utcnow(),
        }
        result = db.test_events.insert_one(test_event)
        print(f"✅ Test insert successful (ID: {result.inserted_id})")

        # Cleanup
        db.test_events.delete_one({"_id": result.inserted_id})

        client.close()
        return True

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\n📝 Troubleshooting steps:")
        print("   1. Verify MongoDB URI is correct in .env")
        print("   2. Check MongoDB Atlas > Database Access > Users")
        print("      - User 'cctv' should exist with password from .env")
        print("   3. Check MongoDB Atlas > Network Access")
        print("      - Your IP address should be whitelisted (or add 0.0.0.0/0 for dev)")
        print("   4. Verify database name matches: MONGODB_DB_NAME in .env")
        return False


def show_setup_instructions():
    """Show MongoDB Atlas setup instructions."""
    print("\n" + "="*60)
    print("MongoDB Atlas Setup Instructions")
    print("="*60)
    print("""
1. Go to MongoDB Atlas (https://cloud.mongodb.com)
2. Select your cluster
3. Click "Connect" → "Connection String"
4. Copy the connection string: mongodb+srv://cctv:PASSWORD@...

5. Update .env file:
   MONGODB_URI=<paste-connection-string>
   MONGODB_DB_NAME=cctv

6. In MongoDB Atlas, verify:
   - Database Access: User 'cctv' with correct password
   - Network Access: Your IP is whitelisted
   - Collections: Should auto-create on first insert

7. Run this script to test:
   python3 setup_mongodb.py
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        show_setup_instructions()
    else:
        success = test_connection()
        if not success:
            print("\n💡 Run with --setup flag for instructions:")
            print("   python3 setup_mongodb.py --setup")
        sys.exit(0 if success else 1)
