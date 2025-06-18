//! Autobahn Command Line Interface
//!
//! This CLI provides comprehensive management and interaction capabilities
//! for the Autobahn biological metabolism computer system.

use autobahn::*;
use clap::{App, Arg, SubCommand};
use std::process;
use tokio;

#[tokio::main]
async fn main() {
    let matches = App::new("Autobahn CLI")
        .version(autobahn::version())
        .author("Kundai Farai Sachikonye <kundai.f.sachikonye@gmail.com>")
        .about("Command line interface for the Autobahn biological metabolism computer")
        .subcommand(
            SubCommand::with_name("init")
                .about("Initialize Autobahn system")
                .arg(
                    Arg::with_name("config")
                        .short("c")
                        .long("config")
                        .value_name("FILE")
                        .help("Configuration file path")
                        .takes_value(true)
                )
        )
        .subcommand(
            SubCommand::with_name("process")
                .about("Process information through Autobahn")
                .arg(
                    Arg::with_name("input")
                        .short("i")
                        .long("input")
                        .value_name("TEXT")
                        .help("Input text to process")
                        .takes_value(true)
                        .required(true)
                )
                .arg(
                    Arg::with_name("mode")
                        .short("m")
                        .long("mode")
                        .value_name("MODE")
                        .help("Processing mode (quick, comprehensive)")
                        .takes_value(true)
                        .default_value("quick")
                )
        )
        .subcommand(
            SubCommand::with_name("benchmark")
                .about("Run system benchmarks")
                .arg(
                    Arg::with_name("type")
                        .short("t")
                        .long("type")
                        .value_name("TYPE")
                        .help("Benchmark type (performance, atp, modules, stress)")
                        .takes_value(true)
                        .default_value("performance")
                )
                .arg(
                    Arg::with_name("iterations")
                        .short("n")
                        .long("iterations")
                        .value_name("COUNT")
                        .help("Number of iterations")
                        .takes_value(true)
                        .default_value("10")
                )
        )
        .subcommand(
            SubCommand::with_name("status")
                .about("Show system status and health")
        )
        .subcommand(
            SubCommand::with_name("config")
                .about("Configuration management")
                .subcommand(
                    SubCommand::with_name("show")
                        .about("Show current configuration")
                )
                .subcommand(
                    SubCommand::with_name("generate")
                        .about("Generate default configuration file")
                        .arg(
                            Arg::with_name("output")
                                .short("o")
                                .long("output")
                                .value_name("FILE")
                                .help("Output file path")
                                .takes_value(true)
                                .default_value("autobahn.toml")
                        )
                )
        )
        .subcommand(
            SubCommand::with_name("plugins")
                .about("Plugin management")
                .subcommand(
                    SubCommand::with_name("list")
                        .about("List installed plugins")
                )
                .subcommand(
                    SubCommand::with_name("status")
                        .about("Show plugin status")
                        .arg(
                            Arg::with_name("plugin")
                                .value_name("PLUGIN_ID")
                                .help("Plugin ID")
                                .takes_value(true)
                                .required(true)
                        )
                )
        )
        .get_matches();

    // Initialize system
    if let Err(e) = autobahn::init() {
        eprintln!("Failed to initialize Autobahn: {}", e);
        process::exit(1);
    }

    let result = match matches.subcommand() {
        ("init", Some(sub_matches)) => handle_init(sub_matches).await,
        ("process", Some(sub_matches)) => handle_process(sub_matches).await,
        ("benchmark", Some(sub_matches)) => handle_benchmark(sub_matches).await,
        ("status", Some(_)) => handle_status().await,
        ("config", Some(sub_matches)) => handle_config(sub_matches).await,
        ("plugins", Some(sub_matches)) => handle_plugins(sub_matches).await,
        _ => {
            eprintln!("No subcommand provided. Use --help for usage information.");
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

async fn handle_init(matches: &clap::ArgMatches<'_>) -> Result<(), AutobahnError> {
    println!("🧬 Initializing Autobahn Biological Metabolism Computer");
    
    let config_path = matches.value_of("config").unwrap_or("autobahn.toml");
    
    let mut config_manager = ConfigurationManager::new();
    config_manager.load_from_file(config_path)?;
    
    let mut system = AutobahnSystem::new();
    system.initialize().await?;
    
    println!("✅ Autobahn system initialized successfully");
    println!("📊 System capabilities:");
    
    let capabilities = get_capabilities();
    println!("   • V8 Modules: {:?}", capabilities.available_modules);
    println!("   • Processing modes: {:?}", capabilities.processing_modes);
    println!("   • Max ATP capacity: {:.1}", capabilities.max_atp_capacity);
    
    Ok(())
}

async fn handle_process(matches: &clap::ArgMatches<'_>) -> Result<(), AutobahnError> {
    let input = matches.value_of("input").unwrap();
    let mode = matches.value_of("mode").unwrap();
    
    println!("🔬 Processing input through Autobahn system");
    println!("Input: {}", input);
    println!("Mode: {}", mode);
    
    let start_time = std::time::Instant::now();
    
    let result = match mode {
        "quick" => {
            let result = quick_process(input).await?;
            println!("\n📊 Quick Processing Results:");
            println!("   • Confidence: {:.3}", result.confidence);
            println!("   • ATP consumed: {:.2}", result.atp_consumed);
            println!("   • Processing time: {} ms", result.processing_time_ms);
            println!("   • Modules activated: {:?}", result.modules_activated);
            result.content
        }
        "comprehensive" => {
            let result = quick_comprehensive_process(input).await?;
            println!("\n📊 Comprehensive Processing Results:");
            println!("   • Confidence: {:.3}", result.processing_metadata.confidence_score);
            println!("   • ATP consumed: {:.2}", result.processing_metadata.total_atp_consumed);
            println!("   • Processing time: {} ms", result.processing_metadata.processing_time_ms);
            println!("   • Modules used: {:?}", result.processing_metadata.modules_used);
            
            if let Some(uncertainty) = &result.uncertainty_analysis {
                println!("   • Uncertainty level: {:.3}", uncertainty.uncertainty_level);
                println!("   • Confidence intervals: {} intervals", uncertainty.confidence_intervals.len());
            }
            
            println!("   • Temporal insights: {} patterns", result.temporal_insights.len());
            
            result.biological_result.content
        }
        _ => {
            return Err(AutobahnError::ProcessingError {
                layer: "cli".to_string(),
                reason: format!("Unknown processing mode: {}", mode),
            });
        }
    };
    
    let elapsed = start_time.elapsed();
    
    println!("\n🧠 Processing Output:");
    println!("{}", result);
    
    println!("\n⏱️  Total execution time: {:.2} ms", elapsed.as_millis());
    
    Ok(())
}

async fn handle_benchmark(matches: &clap::ArgMatches<'_>) -> Result<(), AutobahnError> {
    let benchmark_type = matches.value_of("type").unwrap();
    let iterations: u32 = matches.value_of("iterations").unwrap().parse()
        .map_err(|_| AutobahnError::ProcessingError {
            layer: "cli".to_string(),
            reason: "Invalid iterations count".to_string(),
        })?;
    
    println!("⚡ Running Autobahn benchmarks");
    println!("Type: {}", benchmark_type);
    println!("Iterations: {}", iterations);
    
    let mut benchmark_config = benchmarking::BenchmarkConfig::default();
    benchmark_config.iterations_per_benchmark = iterations;
    
    match benchmark_type {
        "performance" => benchmark_config.enable_performance = true,
        "atp" => benchmark_config.enable_atp_tests = true,
        "modules" => benchmark_config.enable_module_comparison = true,
        "stress" => benchmark_config.enable_stress_tests = true,
        _ => {
            return Err(AutobahnError::ProcessingError {
                layer: "cli".to_string(),
                reason: format!("Unknown benchmark type: {}", benchmark_type),
            });
        }
    }
    
    let mut benchmark_suite = AutobahnBenchmarkSuite::with_config(benchmark_config);
    benchmark_suite.add_standard_benchmarks();
    
    let mut system = AutobahnSystem::new();
    system.initialize().await?;
    
    println!("🏃 Running benchmarks...");
    benchmark_suite.run_all_benchmarks(&mut system).await?;
    
    let results = benchmark_suite.get_results();
    
    println!("\n📊 Benchmark Results:");
    println!("   • Performance tests: {}", results.benchmark_results.len());
    println!("   • ATP efficiency tests: {}", results.atp_efficiency_results.len());
    println!("   • Module comparison tests: {}", results.module_comparison_results.len());
    println!("   • Stress tests: {}", results.stress_test_results.len());
    
    // Show detailed results
    for result in &results.benchmark_results {
        let status = if result.passed { "✅ PASSED" } else { "❌ FAILED" };
        println!("   {} {}: {:.2} ms avg, {:.2} ops/sec", 
                 status, result.benchmark_name, 
                 result.execution_stats.avg_execution_time_ms,
                 result.performance_metrics.throughput);
    }
    
    if !results.optimization_recommendations.is_empty() {
        println!("\n💡 Optimization Recommendations:");
        for (i, rec) in results.optimization_recommendations.iter().enumerate() {
            println!("   {}. {} (Priority: {:?})", i + 1, rec.description, rec.priority);
            println!("      Expected improvement: {:.1}%", rec.estimated_improvement_percent);
        }
    }
    
    println!("\n📄 Full report generated: {}", benchmark_suite.generate_performance_report().len());
    
    Ok(())
}

async fn handle_status() -> Result<(), AutobahnError> {
    println!("🏥 Autobahn System Status");
    
    let mut system = AutobahnSystem::new();
    system.initialize().await?;
    
    // System health
    let health_report = system.health_check().await?;
    println!("\n🔋 System Health: {:?}", health_report.overall_health);
    println!("   • ATP Status: {:.2}/{:.2}", 
             health_report.atp_status.current_atp, 
             health_report.atp_status.max_atp);
    println!("   • System uptime: {} ms", health_report.system_uptime_ms);
    
    if let Some(error) = &health_report.last_error {
        println!("   • Last error: {}", error);
    }
    
    // Module status
    println!("\n🧩 Module Status:");
    for module_status in &health_report.module_status {
        println!("   • {}: {:?} (ATP rate: {:.2}/s)", 
                 module_status.module_name, 
                 module_status.status,
                 module_status.atp_consumption_rate);
    }
    
    // System monitor
    let monitor = system.get_monitor();
    println!("\n📊 Performance Metrics:");
    println!("   • Average processing time: {:.2} ms", monitor.performance_metrics.avg_processing_time_ms);
    println!("   • Total operations: {}", monitor.performance_metrics.total_operations);
    println!("   • ATP efficiency: {:.2} ops/ATP", monitor.performance_metrics.atp_efficiency);
    println!("   • Error rate: {:.3}%", monitor.performance_metrics.error_rate * 100.0);
    println!("   • Throughput: {:.2} ops/sec", monitor.performance_metrics.throughput);
    
    println!("\n💾 Resource Usage:");
    println!("   • Current ATP: {:.2}", monitor.resource_usage.current_atp);
    println!("   • Peak ATP usage: {:.2}", monitor.resource_usage.peak_atp_usage);
    println!("   • Memory usage: {:.2} MB", monitor.resource_usage.memory_usage_mb);
    println!("   • CPU usage: {:.1}%", monitor.resource_usage.cpu_usage_percent);
    println!("   • Active modules: {}", monitor.resource_usage.active_modules);
    
    // System capabilities
    let capabilities = get_capabilities();
    println!("\n🚀 System Capabilities:");
    println!("   • Probabilistic processing: {}", capabilities.supports_probabilistic);
    println!("   • Adversarial testing: {}", capabilities.supports_adversarial);
    println!("   • Champagne phase: {}", capabilities.supports_champagne);
    println!("   • Available modules: {}", capabilities.available_modules.len());
    println!("   • Processing modes: {}", capabilities.processing_modes.len());
    
    Ok(())
}

async fn handle_config(matches: &clap::ArgMatches<'_>) -> Result<(), AutobahnError> {
    match matches.subcommand() {
        ("show", Some(_)) => {
            println!("📋 Current Autobahn Configuration");
            
            let mut config_manager = ConfigurationManager::new();
            config_manager.load_from_file("autobahn.toml").unwrap_or_else(|_| {
                println!("No config file found, showing defaults");
            });
            
            let config = config_manager.get_config();
            
            println!("\n🔧 System Configuration:");
            println!("   • Name: {}", config.system.name);
            println!("   • Version: {}", config.system.version);
            println!("   • Environment: {:?}", config.system.environment);
            println!("   • Max ATP capacity: {:.1}", config.system.max_atp_capacity);
            println!("   • ATP regeneration rate: {:.2}", config.system.atp_regeneration_rate);
            println!("   • Debug mode: {}", config.system.debug_mode);
            println!("   • Thread pool size: {}", config.system.thread_pool_size);
            
            println!("\n🧬 V8 Pipeline Configuration:");
            println!("   • Enabled: {}", config.v8_pipeline.enabled);
            println!("   • Modules: {}", config.v8_pipeline.modules.len());
            println!("   • ATP allocation: {:?}", config.v8_pipeline.atp_allocation_strategy);
            println!("   • Processing timeout: {} ms", config.v8_pipeline.processing_timeout_ms);
            println!("   • Parallel processing: {}", config.v8_pipeline.enable_parallel_processing);
            
            println!("\n🎲 Probabilistic Configuration:");
            println!("   • Enabled: {}", config.probabilistic.enabled);
            println!("   • Inference algorithm: {}", config.probabilistic.default_inference_algorithm);
            println!("   • Monte Carlo samples: {}", config.probabilistic.monte_carlo.default_samples);
            
            println!("\n⏰ Temporal Configuration:");
            println!("   • Enabled: {}", config.temporal.enabled);
            println!("   • Decay function: {}", config.temporal.decay_function);
            println!("   • Decay rate: {:.3}", config.temporal.decay_rate);
            println!("   • Historical retention: {} days", config.temporal.historical_retention_days);
            
            println!("\n🔬 Research Configuration:");
            println!("   • Enabled: {}", config.research.enabled);
            println!("   • Quantum processing: {}", config.research.enable_quantum);
            println!("   • ML integration: {}", config.research.enable_ml);
            println!("   • Experimental pathways: {}", config.research.enable_experimental);
        }
        ("generate", Some(sub_matches)) => {
            let output_path = sub_matches.value_of("output").unwrap();
            
            println!("📝 Generating default configuration file: {}", output_path);
            
            ConfigurationManager::generate_default_config_file(output_path)?;
            
            println!("✅ Default configuration file generated successfully");
            println!("📄 Edit the file to customize your Autobahn system settings");
        }
        _ => {
            eprintln!("No config subcommand provided. Use 'config --help' for usage.");
        }
    }
    
    Ok(())
}

async fn handle_plugins(matches: &clap::ArgMatches<'_>) -> Result<(), AutobahnError> {
    let plugin_config = plugins::PluginManagerConfig::default();
    let plugin_manager = PluginManager::new(plugin_config);
    
    match matches.subcommand() {
        ("list", Some(_)) => {
            println!("🔌 Installed Plugins");
            
            let plugins = plugin_manager.list_plugins();
            if plugins.is_empty() {
                println!("   No plugins installed");
            } else {
                for plugin in plugins {
                    println!("   • {} v{} by {}", plugin.name, plugin.version, plugin.author);
                    println!("     ID: {}", plugin.id);
                    println!("     Description: {}", plugin.description);
                    println!("     Capabilities: {:?}", plugin.capabilities);
                    println!("     Dependencies: {:?}", plugin.dependencies);
                    println!();
                }
            }
        }
        ("status", Some(sub_matches)) => {
            let plugin_id = sub_matches.value_of("plugin").unwrap();
            
            println!("🔍 Plugin Status: {}", plugin_id);
            
            if let Some(status) = plugin_manager.get_plugin_status(plugin_id) {
                println!("   • Status: {:?}", status);
                
                match plugin_manager.get_plugin_health(plugin_id).await {
                    Ok(health) => {
                        println!("   • Health: {}", if health.healthy { "✅ Healthy" } else { "❌ Unhealthy" });
                        println!("   • Health score: {:.2}", health.health_score);
                        println!("   • Last check: {}", health.last_check);
                        
                        for check in &health.checks {
                            let status = if check.passed { "✅" } else { "❌" };
                            println!("     {} {}: {} ({} ms)", status, check.name, check.message, check.duration_ms);
                        }
                    }
                    Err(e) => {
                        println!("   • Health check failed: {}", e);
                    }
                }
            } else {
                println!("   Plugin not found or not loaded");
            }
        }
        _ => {
            eprintln!("No plugins subcommand provided. Use 'plugins --help' for usage.");
        }
    }
    
    Ok(())
} 