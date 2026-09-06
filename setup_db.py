from src.data import load_demo_data, LOT_META, BUILDINGS
from src.database import get_engine, init_schema, seed_database

def main():
    engine = get_engine()
    if engine is None:
        raise SystemExit(
            "DATABASE_URL is not set.\n"
            "Example:\n"
            "postgresql+psycopg2://postgres:password@localhost:5432/smartpark"
        )

    print("Creating schema...")
    init_schema(engine)

    print("Loading university parking dataset...")
    history = load_demo_data()

    print(f"Seeding {len(history):,} occupancy observations...")
    seed_database(engine, LOT_META, BUILDINGS, history)

    print("Database setup complete.")

if __name__ == "__main__":
    main()
