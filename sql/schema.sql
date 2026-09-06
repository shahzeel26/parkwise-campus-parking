CREATE TABLE IF NOT EXISTS parking_lots (
    lot_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    capacity INT NOT NULL CHECK (capacity > 0),
    permit_type VARCHAR(40),
    hourly_rate DECIMAL(8,2) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS campus_buildings (
    building_id SERIAL PRIMARY KEY,
    building_name VARCHAR(150) UNIQUE NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL
);

CREATE TABLE IF NOT EXISTS occupancy_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    lot_id VARCHAR(20) REFERENCES parking_lots(lot_id),
    observed_at TIMESTAMP NOT NULL,
    occupied_spaces INT NOT NULL CHECK (occupied_spaces >= 0),
    available_spaces INT NOT NULL CHECK (available_spaces >= 0),
    occupancy_rate DECIMAL(6,5) NOT NULL CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    rain_mm DECIMAL(7,2) DEFAULT 0,
    event_flag BOOLEAN DEFAULT FALSE,
    exam_period BOOLEAN DEFAULT FALSE,
    UNIQUE(lot_id, observed_at)
);

CREATE TABLE IF NOT EXISTS occupancy_forecasts (
    forecast_id BIGSERIAL PRIMARY KEY,
    lot_id VARCHAR(20) REFERENCES parking_lots(lot_id),
    forecast_for TIMESTAMP NOT NULL,
    predicted_occupied INT NOT NULL,
    predicted_available INT NOT NULL,
    predicted_occupancy_rate DECIMAL(6,5) NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_occupancy_lot_time
ON occupancy_observations(lot_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_forecast_lot_time
ON occupancy_forecasts(lot_id, forecast_for);
