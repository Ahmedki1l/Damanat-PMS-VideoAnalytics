-- SQL Schema for Config and Preprocessing tables
-- Optimized for SQL Server (T-SQL)

-- 1. Create 'config' table
CREATE TABLE [config] (
    [id] INT IDENTITY(1,1) PRIMARY KEY,
    
    -- Processing settings
    [target_fps] INT NOT NULL DEFAULT 20,
    [processing_mode] VARCHAR(50) NOT NULL DEFAULT 'round_robin', -- Enum: round_robin, priority, single_stream
    [stream_channel] INT NOT NULL DEFAULT 102,                  -- Enum: 101 (Main), 102 (Sub)
    
    -- Slot reference resolution
    [slot_ref_width] INT NOT NULL DEFAULT 640,
    [slot_ref_height] INT NOT NULL DEFAULT 360,
    
    -- Detector settings
    [device] VARCHAR(20) NOT NULL DEFAULT 'cpu',                -- Enum: cpu, cuda
    [model_path] VARCHAR(255) NOT NULL,
    [confidence] FLOAT NOT NULL DEFAULT 0.20,
    [imgsz] INT NOT NULL DEFAULT 640,
    [classes] VARCHAR(50) NOT NULL DEFAULT '2,5,7',             -- Comma-separated list of COCO IDs
    
    -- Tracker settings
    [tracker_type] VARCHAR(50) NOT NULL DEFAULT 'bytetrack',    -- Enum: bytetrack, deepsort
    
    -- State Machine settings
    [confirm_enter_frames] INT NOT NULL DEFAULT 10,
    [confirm_leave_frames] INT NOT NULL DEFAULT 40,
    
    -- Assigner settings
    [overlap_threshold] FLOAT NOT NULL DEFAULT 0.3,
    
    -- Output settings
    [show_video] BIT NOT NULL DEFAULT 0,
    [show_camera] VARCHAR(50) NULL,
    [log_file] VARCHAR(255) NULL
);

-- 2. Create 'preprocessing_config' table
CREATE TABLE [preprocessing_config] (
    [id] INT IDENTITY(1,1) PRIMARY KEY,
    [config_id] INT NOT NULL,
    
    [scope] VARCHAR(20) NOT NULL,                               -- Enum: detector, reid
    [enabled] BIT NOT NULL DEFAULT 1,
    
    [clip_limit] FLOAT NOT NULL DEFAULT 2.0,
    [grid_w] INT NOT NULL DEFAULT 8,
    [grid_h] INT NOT NULL DEFAULT 8,
    
    [gamma_correction] BIT NULL,
    [dark_threshold] INT NULL,
    
    CONSTRAINT [FK_preprocessing_config_config] FOREIGN KEY ([config_id]) 
        REFERENCES [config] ([id]) ON DELETE CASCADE
);
