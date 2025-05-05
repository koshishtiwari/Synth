import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import traceback

from src.config.config_manager import config_manager
from src.core.engine import SynthEngine
from src.models.data_generators import TabularDataGenerator, TimeSeriesGenerator
from src.core.streaming import ConsoleStreamer, FileStreamer, KafkaStreamer

# Define API data models
class GeneratorSchema(BaseModel):
    name: str
    type: str
    schema: Dict[str, Any]

class StreamerConfig(BaseModel):
    name: str
    type: str
    config: Dict[str, Any]

class EngineConfig(BaseModel):
    batch_size: Optional[int] = None
    generation_frequency: Optional[float] = None
    drift_enabled: Optional[bool] = None
    drift_interval: Optional[int] = None
    drift_magnitude: Optional[float] = None

# Global engine instance
engine = SynthEngine()

# Create FastAPI app
app = FastAPI(
    title="Synth API",
    description="API for synthetic data generation system",
    version="0.1.0",
)

@app.on_event("startup")
def startup_event():
    """Initialize resources on startup."""
    logging.basicConfig(
        level=getattr(logging, config_manager.get('logging.level', 'INFO')),
        format=config_manager.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    )

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown."""
    engine.close()

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to Synth API", "status": "active"}

@app.get("/status")
def get_status():
    """Get system status."""
    return {
        "running": engine.running,
        "generators": list(engine.generators.keys()),
        "streamers": list(engine.streaming_manager.streamers.keys()),
        "stats": engine.get_stats(),
    }

@app.post("/generators")
def add_generator(generator_schema: GeneratorSchema):
    """Add a new data generator."""
    try:
        if generator_schema.type.lower() == "tabular":
            generator = TabularDataGenerator(generator_schema.name, generator_schema.schema)
        elif generator_schema.type.lower() == "timeseries":
            generator = TimeSeriesGenerator(generator_schema.name, generator_schema.schema)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported generator type: {generator_schema.type}")
        
        engine.add_generator(generator)
        return {"message": f"Generator {generator_schema.name} added successfully"}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generators")
def list_generators():
    """List all data generators."""
    return {
        "generators": [
            {
                "name": name,
                "metadata": generator.metadata()
            }
            for name, generator in engine.generators.items()
        ]
    }

@app.delete("/generators/{name}")
def remove_generator(name: str):
    """Remove a data generator."""
    if name not in engine.generators:
        raise HTTPException(status_code=404, detail=f"Generator {name} not found")
    
    engine.remove_generator(name)
    return {"message": f"Generator {name} removed successfully"}

@app.post("/streamers")
def add_streamer(streamer_config: StreamerConfig):
    """Add a new data streamer."""
    try:
        streamer_type = streamer_config.type.lower()
        config = streamer_config.config
        
        if streamer_type == "console":
            streamer = ConsoleStreamer(
                streamer_config.name,
                pretty_print=config.get("pretty_print", True)
            )
        elif streamer_type == "file":
            if "file_path" not in config:
                raise HTTPException(status_code=400, detail="file_path is required for file streamer")
            
            streamer = FileStreamer(
                streamer_config.name,
                config["file_path"],
                format=config.get("format", "csv"),
                mode=config.get("mode", "a")
            )
        elif streamer_type == "kafka":
            if "topic" not in config:
                raise HTTPException(status_code=400, detail="topic is required for Kafka streamer")
            
            streamer = KafkaStreamer(
                streamer_config.name,
                config["topic"],
                config.get("bootstrap_servers", "localhost:9092")
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported streamer type: {streamer_type}")
        
        engine.add_streamer(streamer)
        return {"message": f"Streamer {streamer_config.name} added successfully"}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/streamers")
def list_streamers():
    """List all data streamers."""
    return {
        "streamers": [
            {
                "name": name,
                "metadata": streamer.metadata()
            }
            for name, streamer in engine.streaming_manager.streamers.items()
        ]
    }

@app.delete("/streamers/{name}")
def remove_streamer(name: str):
    """Remove a data streamer."""
    if name not in engine.streaming_manager.streamers:
        raise HTTPException(status_code=404, detail=f"Streamer {name} not found")
    
    engine.remove_streamer(name)
    return {"message": f"Streamer {name} removed successfully"}

@app.post("/generate")
def generate_data(apply_drift: bool = False, generator_name: Optional[str] = None):
    """Generate a single batch of data."""
    if not engine.generators:
        raise HTTPException(status_code=400, detail="No generators configured")
    
    if not engine.streaming_manager.streamers:
        raise HTTPException(status_code=400, detail="No streamers configured")
    
    results = engine.run_once(apply_drift=apply_drift)
    return {
        "message": "Data generated and streamed successfully",
        "results": results
    }

@app.post("/start")
def start_continuous_generation(background_tasks: BackgroundTasks):
    """Start continuous data generation."""
    if not engine.generators:
        raise HTTPException(status_code=400, detail="No generators configured")
    
    if not engine.streaming_manager.streamers:
        raise HTTPException(status_code=400, detail="No streamers configured")
    
    if engine.running:
        return {"message": "Engine is already running"}
    
    # Start in background to not block the API
    background_tasks.add_task(engine.run_continuous)
    return {"message": "Continuous data generation started"}

@app.post("/stop")
def stop_continuous_generation():
    """Stop continuous data generation."""
    if not engine.running:
        return {"message": "Engine is not running"}
    
    engine.stop()
    return {"message": "Continuous data generation stopped"}

@app.post("/config")
def update_configuration(config: EngineConfig):
    """Update engine configuration."""
    engine_config = {}
    
    if config.batch_size is not None:
        engine_config["batch_size"] = config.batch_size
        
    if config.generation_frequency is not None:
        engine_config["generation_frequency"] = config.generation_frequency
        
    if config.drift_enabled is not None:
        engine_config["drift_enabled"] = config.drift_enabled
        
    if config.drift_interval is not None:
        engine_config["drift_interval"] = config.drift_interval
        
    if config.drift_magnitude is not None:
        engine_config["drift_magnitude"] = config.drift_magnitude
    
    engine.configure(engine_config)
    return {"message": "Configuration updated successfully"}

def run_api_server():
    """Run the API server."""
    host = config_manager.get('api.host', '0.0.0.0')
    port = config_manager.get('api.port', 8000)
    
    uvicorn.run(app, host=host, port=port)