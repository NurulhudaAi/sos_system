#!/usr/bin/env python3
"""MongoDB setup and connection test for CCTV SOS System."""

import sys
import os
from pathlib import Path
import re

# Ensure env is loaded
from config.env_manager import init_env
init_env(require_edit=False)


def mask_mongodb_uri(uri: str) -> str:
    """Mask credentials in MongoDB URI for console output."""
    if not uri:
        return "(empty)"
    masked = re.sub(r"(mongodb(?:\+srv)?://[^:/@]+:)[^@]+(@)", r"\1***\2", uri)
    return masked if len(masked) <= 80 else masked[:77] + "..."

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

    print(f"Testing connection to: {mask_mongodb_uri(uri)}")
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

        # Test app incident insert path
        print("\n⏳ Testing cctv_incidents insert operation...")
        import uuid
        event_uuid = "setup-test-" + str(uuid.uuid4())
        database.insert_sos_event(
            event_uuid=event_uuid,
            event_type="setup_test",
            severity=0,
            severity_name="LOG",
            source_id="setup_mongodb",
            location="setup",
            extra={"setup_test": True},
        )
        inserted = database.get_event_by_uuid(event_uuid)
        if not inserted:
            raise RuntimeError("Inserted setup test incident was not found")
        print(f"✅ Incident insert successful (UUID: {event_uuid})")

        # Cleanup
        db[database.INCIDENTS_COLLECTION].delete_one({"event_uuid": event_uuid})

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
    db_name = os.getenv("MONGODB_DB_NAME", "iam")
    print("\n" + "="*60)
    print("MongoDB Atlas Setup Instructions")
    print("="*60)
    print(f"""
1. Go to MongoDB Atlas (https://cloud.mongodb.com)
2. Select your cluster
3. Click "Connect" → "Connection String"
4. Copy the connection string: mongodb+srv://USERNAME:PASSWORD@...

5. Update .env file:
   MONGODB_URI=<paste-connection-string>
   MONGODB_DB_NAME={db_name}

6. In MongoDB Atlas, verify:
   - Database Access: USERNAME from MONGODB_URI exists with correct password
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
