
"""
Configuration
"""

import os 
from dataclasses import dataclass ,field 
from typing import Dict ,List ,Any ,Optional 
from pathlib import Path 

@dataclass 
class MAMAConfig :

    system_name :str ="MAMA_Framework"
    random_seed :int =42 

    data_dir :str ="data"
    models_dir :str ="models"
    logs_dir :str ="logs"
    results_dir :str ="results"
    cache_dir :str ="cache"

    trust_weights :Dict [str ,float ]=field (default_factory =lambda :{
    "reliability":0.25 ,
    "competence":0.20 ,
    "fairness":0.15 ,
    "security":0.20 ,
    "transparency":0.20 
    })

    trust_threshold :float =0.3 
    confidence_threshold :float =0.8 
    score_decay_factor :float =0.95 
    confidence_threshold_trust :float =0.7 

    hash_algorithm :str ="sha256"
    hash_length :int =16 

    selection_alpha :float =0.2 

    max_concurrent_agents :int =5 
    max_concurrent_tasks :int =10 
    default_task_timeout :float =30.0 
    timeout_seconds :float =30.0 
    consensus_threshold :float =0.75 
    similarity_threshold :float =0.6 

    sbert_model_name :str ="all-MiniLM-L6-v2"
    embedding_dimension :int =384 
    similarity_metric :str ="cosine"

    enable_embedding_cache :bool =True 
    cache_expiry_hours :int =24 

    learning_rate :float =0.001 
    discount_factor :float =0.95 
    epsilon_start :float =1.0 
    epsilon_end :float =0.01 
    epsilon_decay :float =0.995 

    trust_weight :float =0.4 
    exploration_strategy :str ="epsilon_greedy"

    state_dimension :int =128 
    action_dimension :int =64 
    marl_state_size :int =128 

    feature_dimension :int =128 
    hidden_dimensions :List [int ]=field (default_factory =lambda :[256 ,128 ,64 ,32 ])
    ranking_algorithm :str ="listwise"

    ltr_learning_rate :float =0.001 
    batch_size :int =64 
    num_epochs :int =200 
    device :str ="cpu"

    ranking_depth :int =10 
    ndcg_k :int =5 

    reward_lambda1 :float =0.4 
    reward_lambda2 :float =0.4 
    reward_lambda3 :float =0.2 

    agent_capabilities :Dict [str ,Dict [str ,Any ]]=field (default_factory =lambda :{
    "weather_agent":{
    "specialty":"meteorological analysis and atmospheric safety assessment",
    "expertise_areas":["weather_forecasting","atmospheric_conditions","safety_meteorology"],
    "input_types":["departure_location","destination_location","flight_time","route_coordinates"],
    "output_types":["safety_score","weather_conditions","meteorological_report"],
    "trust_dimensions":["reliability","competence","transparency","predictive_accuracy"],
    "computational_complexity":"high",
    "response_time_sla":2.0 ,
    "accuracy_requirements":0.95 
    },
    "safety_assessment_agent":{
    "specialty":"comprehensive aviation safety evaluation and risk analysis",
    "expertise_areas":["aviation_safety","risk_assessment","airline_analysis","airport_security"],
    "input_types":["weather_safety_score","airline_data","airport_ratings","aircraft_specifications"],
    "output_types":["overall_safety_score","risk_factors","safety_assessment_report"],
    "trust_dimensions":["reliability","competence","fairness","security","precision"],
    "computational_complexity":"very_high",
    "response_time_sla":3.0 ,
    "accuracy_requirements":0.98 
    },
    "flight_info_agent":{
    "specialty":"aviation data retrieval and flight information processing",
    "expertise_areas":["flight_data","schedule_optimization","route_analysis","availability_tracking"],
    "input_types":["departure","destination","date_range","passenger_count"],
    "output_types":["flight_list","availability_status","schedule_optimization"],
    "trust_dimensions":["reliability","competence","transparency","timeliness"],
    "computational_complexity":"medium",
    "response_time_sla":1.5 ,
    "accuracy_requirements":0.92 
    },
    "economic_agent":{
    "specialty":"comprehensive financial analysis and cost optimization",
    "expertise_areas":["cost_analysis","pricing_optimization","economic_forecasting","budget_allocation"],
    "input_types":["flight_list","user_preferences","market_data","economic_indicators"],
    "output_types":["total_cost_per_flight","cost_breakdown","economic_analysis"],
    "trust_dimensions":["competence","fairness","transparency","economic_accuracy"],
    "computational_complexity":"high",
    "response_time_sla":2.5 ,
    "accuracy_requirements":0.94 
    },
    "ltr_ranking_engine":{
    "specialty":"multi-dimensional decision integration and recommendation synthesis",
    "expertise_areas":["decision_integration","multi_criteria_optimization","preference_alignment","recommendation_synthesis"],
    "input_types":["safety_scores","cost_data","user_preferences","optimization_constraints"],
    "output_types":["ranked_flight_recommendations","explanation","confidence_intervals"],
    "trust_dimensions":["reliability","competence","fairness","transparency","integration_quality"],
    "computational_complexity":"very_high",
    "response_time_sla":4.0 ,
    "accuracy_requirements":0.96 
    }
    })

    evaluation_metrics :List [str ]=field (default_factory =lambda :[
    "MRR",
    "NDCG@5",
    "ART",
    "precision",
    "recall",
    "f1_score"
    ])

    enable_performance_monitoring :bool =True 
    monitoring_interval_seconds :int =60 

    weather_api_key :str =field (default_factory =lambda :os .getenv ("WEATHER_API_KEY",""))
    flight_api_key :str =field (default_factory =lambda :os .getenv ("FLIGHT_API_KEY",""))

    milestone_url :str ="http://localhost:1026"
    protected_url :str ="http://localhost:6003"
    context_url :str ="http://localhost:8080"
    mcp_server_url :str ="ws://localhost:8765"

    dataset_size :int =150 
    train_split :float =0.7 
    validation_split :float =0.15 
    test_split :float =0.15 

    num_experiment_runs :int =5 
    enable_ablation_studies :bool =True 

    log_level :str ="INFO"
    enable_file_logging :bool =True 
    enable_console_logging :bool =True 
    log_format :str ="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    component_log_levels :Dict [str ,str ]=field (default_factory =lambda :{
    "core.marl_system":"DEBUG",
    "core.sbert_similarity":"DEBUG",
    "core.ltr_ranker":"DEBUG",
    "core.multi_dimensional_trust_ledger":"INFO",
    "agents":"INFO",
    "orchestration":"INFO"
    })

    def __post_init__ (self ):

        trust_sum =sum (self .trust_weights .values ())
        if abs (trust_sum -1.0 )>0.001 :
            raise ValueError (f"Trust weights must sum to 1.0, got {trust_sum }")

        for dir_attr in ['data_dir','models_dir','logs_dir','results_dir','cache_dir']:
            dir_path =Path (getattr (self ,dir_attr ))
            dir_path .mkdir (parents =True ,exist_ok =True )

        if not 0.0 <=self .selection_alpha <=1.0 :
            raise ValueError (f"selection_alpha must be between 0.0 and 1.0, got {self .selection_alpha }")

        lambda_sum =self .reward_lambda1 +self .reward_lambda2 +self .reward_lambda3 
        if abs (lambda_sum -1.0 )>0.001 :
            raise ValueError (f"Reward lambdas should sum to 1.0, got {lambda_sum }")

config =MAMAConfig ()

def get_config ()->MAMAConfig :

    return config 

def update_config (**kwargs )->None :

    global config 
    for key ,value in kwargs .items ():
        if hasattr (config ,key ):
            setattr (config ,key ,value )
        else :
            raise ValueError (f"Unknown configuration parameter: {key }")

if __name__ =="__main__":

    cfg =get_config ()
    print (f"MAMA Configuration loaded successfully")
    print (f"Trust weights: {cfg .trust_weights }")
    print (f"Selection alpha: {cfg .selection_alpha }")
    print (f"Agent capabilities: {len (cfg .agent_capabilities )} agents configured")
