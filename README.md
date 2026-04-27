# Sendspin Time Filter

A Kalman filter-based time synchronization library for maintaining accurate client-server timestamp synchronization with microsecond-level precision.

[![A project from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-project.png)](https://www.openhomefoundation.org/)

## Overview

This library implements a reference two-dimensional Kalman filter that tracks clock offset and drift between client and server using NTP-style time message exchange. It provides adaptive synchronization with automatic recovery from network disruptions and clock adjustments.

## Features

- **NTP-style time synchronization** using 4-timestamp message exchange
- **Kalman filter** for optimal offset and drift estimation
- **Adaptive forgetting factor** for quick recovery from clock adjustments
- **Thread-safe** time conversions and time updates
- **Microsecond-level precision** using double-precision arithmetic

## Core API

```cpp
class SendspinTimeFilter {
 public:
  // Configuration struct (all fields have defaults)
  struct Config {
    double process_std_dev = 0.0;            // Offset random-walk diffusion (µs / sqrt(µs))
    double drift_process_std_dev = 1e-11;    // Drift random-walk diffusion (1 / sqrt(µs); drift itself is dimensionless)
    double forget_factor = 2.0;              // Forgetting factor (>1) for recovery from disruptions
    double adaptive_cutoff = 3.0;            // Multiple of max_error that triggers forgetting
    uint8_t min_samples = 100;               // Minimum samples before adaptive forgetting
    double drift_significance_threshold = 2.0;  // SNR threshold for drift compensation
    double max_error_scale = 0.5;            // Scale applied to max_error before Kalman update
  };

  // Constructors
  explicit SendspinTimeFilter(const Config &config);
  SendspinTimeFilter();  // Uses default Config
  // ...
};

// Update filter with computed offset and uncertainty from NTP exchange
// measurement: ((T2-T1)+(T3-T4))/2 in microseconds
// max_error: ((T4-T1)-(T3-T2))/2 in microseconds
// time_added: Client timestamp when measurement was taken in microseconds
void update(int64_t measurement, int64_t max_error, int64_t time_added);

// Convert between client and server timestamps
int64_t compute_server_time(int64_t client_time) const;
int64_t compute_client_time(int64_t server_time) const;

// Get Kalman offset covariance as a proxy for synchronization accuracy
int64_t get_error() const;
```

## Recommended Values

The default `Config` values are based on preliminary experiments and provide good synchronization performance for typical network conditions:

```cpp
SendspinTimeFilter filter;  // Uses default Config

// Or override specific fields:
SendspinTimeFilter::Config config;
config.forget_factor = 1.5;
SendspinTimeFilter filter(config);
```

The defaults balance tracking responsiveness with stability for typical network conditions.

## Documentation

See [docs/theory.md](docs/theory.md) for detailed mathematical documentation of the Kalman filter implementation and time synchronization protocol.
