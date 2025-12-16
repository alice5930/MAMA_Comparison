
"""
Core Components

It includes:

1. Verifiable Reputation Ledger (VRL) with hash chain support
2. SBERT Similarity Engine for semantic matching
3. Multi-Agent Reinforcement Learning (MARL) System
4. Learning to Rank (LTR) Engine

"""

import json 
import logging 
import hashlib 
import time 
import random 
import pickle 
import math 
import threading 
import uuid 
import os 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from typing import Dict ,List ,Optional ,Any ,Tuple ,Union ,Callable 
from datetime import datetime ,timedelta 
from dataclasses import dataclass ,field ,asdict 
from enum import Enum 
from pathlib import Path 
from collections import defaultdict ,deque ,Counter 
try :
    from sklearn .preprocessing import StandardScaler ,MinMaxScaler 
    from sklearn .model_selection import train_test_split 
    from sklearn .metrics import ndcg_score ,precision_score ,recall_score 
    from sklearn .metrics .pairwise import cosine_similarity 
except ModuleNotFoundError :
    from .sklearn_compat import (
    StandardScaler ,
    MinMaxScaler ,
    train_test_split ,
    ndcg_score ,
    precision_score ,
    recall_score ,
    cosine_similarity 
    )
try :
    from sentence_transformers import SentenceTransformer 
except ModuleNotFoundError :
    from .sentence_transformer_fallback import SentenceTransformer 
from torch .utils .data import Dataset ,DataLoader 
import asyncio 
from config import get_config 

logger =logging .getLogger (__name__ )
config =get_config ()

def sigmoid (x ):
    return 1 /(1 +np .exp (-x ))

class TrustDimension (Enum ):
    RELIABILITY ="reliability"
    COMPETENCE ="competence"
    FAIRNESS ="fairness"
    SECURITY ="security"
    TRANSPARENCY ="transparency"

@dataclass 
class TrustRecord :
    agent_id :str 
    timestamp :datetime 
    dimension :TrustDimension 
    score :float 
    evidence :Dict [str ,Any ]
    evaluator :str 
    transaction_hash :str 
    previous_hash :str =""
    block_index :int =0 

@dataclass 
class DimensionMetrics :
    current_score :float 
    historical_average :float 
    trend :str 
    last_updated :datetime 
    evaluation_count :int 
    confidence_level :float 

class VRL :
    def __init__ (self ):
        self .trust_records :List [TrustRecord ]=[]
        self .dimension_weights =config .trust_weights 
        self .score_decay_factor =config .score_decay_factor 
        self .confidence_threshold =config .confidence_threshold_trust 

        logger .info ("VRL initialized with hash chain support")

    def record_trust_evaluation (self ,agent_id :str ,dimension :TrustDimension ,
    score :float ,evidence :Dict [str ,Any ],
    evaluator :str ="system")->str :
        try :
            if not 0.0 <=score <=1.0 :
                raise ValueError ("Trust score must be between 0.0 and 1.0")

            previous_hash =""
            block_index =0 
            if self .trust_records :
                previous_record =self .trust_records [-1 ]
                previous_hash =previous_record .transaction_hash 
                block_index =previous_record .block_index +1 

            timestamp =datetime .now ()
            hash_input =f"{agent_id }:{dimension .value }:{score }:{timestamp .isoformat ()}:{evaluator }:{previous_hash }:{block_index }"
            transaction_hash =hashlib .sha256 (hash_input .encode ()).hexdigest ()[:config .hash_length ]

            record =TrustRecord (
            agent_id =agent_id ,
            timestamp =timestamp ,
            dimension =dimension ,
            score =score ,
            evidence =evidence ,
            evaluator =evaluator ,
            transaction_hash =transaction_hash ,
            previous_hash =previous_hash ,
            block_index =block_index 
            )

            self .trust_records .append (record )

            logger .info (f"Recorded trust evaluation: {agent_id } - {dimension .value } = {score :.3f} [Block: {block_index }]")
            return transaction_hash 

        except Exception as e :
            logger .error (f"Failed to record trust evaluation: {e }")
            raise 

    def calculate_overall_trust_score (self ,agent_id :str )->Dict [str ,Any ]:
        try :

            agent_records =[r for r in self .trust_records if r .agent_id ==agent_id ]

            if not agent_records :
                return {
                "agent_id":agent_id ,
                "overall_score":0.5 ,
                "trust_level":"unknown",
                "dimension_scores":{},
                "confidence":0.0 ,
                "evaluation_count":0 ,
                "last_updated":None 
                }

            dimension_scores ={}
            current_time =datetime .now ()

            for dimension in TrustDimension :
                dimension_records =[r for r in agent_records if r .dimension ==dimension ]

                if dimension_records :
                    weighted_scores =[]
                    for record in dimension_records :
                        time_diff =(current_time -record .timestamp ).total_seconds ()/3600 
                        decay_factor =self .score_decay_factor **time_diff 
                        weighted_scores .append (record .score *decay_factor )

                    dimension_scores [dimension .value ]=np .mean (weighted_scores )
                else :
                    dimension_scores [dimension .value ]=0.5 

            overall_score =sum (
            self .dimension_weights [dim ]*score 
            for dim ,score in dimension_scores .items ()
            )

            if overall_score >=0.8 :
                trust_level ="high"
            elif overall_score >=0.6 :
                trust_level ="medium"
            elif overall_score >=0.4 :
                trust_level ="low"
            else :
                trust_level ="very_low"

            evaluation_count =len (agent_records )
            confidence =min (1.0 ,evaluation_count /10.0 )

            return {
            "agent_id":agent_id ,
            "overall_score":overall_score ,
            "trust_level":trust_level ,
            "dimension_scores":dimension_scores ,
            "confidence":confidence ,
            "evaluation_count":evaluation_count ,
            "last_updated":max (r .timestamp for r in agent_records ).isoformat (),
            "dimension_weights":self .dimension_weights 
            }

        except Exception as e :
            logger .error (f"Failed to calculate trust score for {agent_id }: {e }")
            return {
            "agent_id":agent_id ,
            "overall_score":0.5 ,
            "trust_level":"error",
            "error":str (e )
            }

    def verify_hash_chain_integrity (self )->Dict [str ,Any ]:
        try :
            if not self .trust_records :
                return {
                "valid":True ,
                "total_records":0 ,
                "message":"Empty ledger - valid by default"
                }

            invalid_records =[]

            for i ,record in enumerate (self .trust_records ):
                if i ==0 :
                    if record .previous_hash !=""or record .block_index !=0 :
                        invalid_records .append ({
                        "index":i ,
                        "issue":"First record should have empty previous_hash and block_index 0",
                        "record_hash":record .transaction_hash 
                        })
                else :
                    previous_record =self .trust_records [i -1 ]
                    if record .previous_hash !=previous_record .transaction_hash :
                        invalid_records .append ({
                        "index":i ,
                        "issue":"Previous hash mismatch",
                        "expected":previous_record .transaction_hash ,
                        "actual":record .previous_hash ,
                        "record_hash":record .transaction_hash 
                        })

                    if record .block_index !=previous_record .block_index +1 :
                        invalid_records .append ({
                        "index":i ,
                        "issue":"Block index not sequential",
                        "expected":previous_record .block_index +1 ,
                        "actual":record .block_index ,
                        "record_hash":record .transaction_hash 
                        })

            is_valid =len (invalid_records )==0 

            return {
            "valid":is_valid ,
            "total_records":len (self .trust_records ),
            "invalid_records":invalid_records ,
            "message":"Hash chain is valid"if is_valid else f"Found {len (invalid_records )} integrity issues"
            }

        except Exception as e :
            logger .error (f"Error verifying hash chain integrity: {e }")
            return {
            "valid":False ,
            "error":str (e ),
            "message":"Verification failed due to error"
            }

    def get_trust_history (self ,agent_id :str ,dimension :Optional [TrustDimension ]=None )->List [Dict [str ,Any ]]:

        try :
            records =[r for r in self .trust_records if r .agent_id ==agent_id ]

            if dimension :
                records =[r for r in records if r .dimension ==dimension ]

            return [
            {
            "timestamp":r .timestamp .isoformat (),
            "dimension":r .dimension .value ,
            "score":r .score ,
            "evaluator":r .evaluator ,
            "transaction_hash":r .transaction_hash ,
            "block_index":r .block_index 
            }
            for r in sorted (records ,key =lambda x :x .timestamp )
            ]

        except Exception as e :
            logger .error (f"Error getting trust history for {agent_id }: {e }")
            return []

@dataclass 
class AEPEntry :
    agent_id :str 
    agent_name :str 
    specialty :str 
    output_type :str 
    capabilities :List [str ]
    expertise_areas :List [str ]
    performance_history :Dict [str ,float ]=field (default_factory =dict )
    last_updated :str =field (default_factory =lambda :datetime .now ().isoformat ())

    def to_dict (self )->Dict [str ,Any ]:
        return {
        "agent":self .agent_name ,
        "specialty":self .specialty ,
        "output_type":self .output_type ,
        "capabilities":self .capabilities ,
        "expertise_areas":self .expertise_areas ,
        "performance_history":self .performance_history ,
        "last_updated":self .last_updated 
        }

    def get_expertise_text (self )->str :
        return f"{self .specialty }. {' '.join (self .capabilities )}. {' '.join (self .expertise_areas )}"

class AEPRepository :
    def __init__ (self ):
        self .entries :Dict [str ,AEPEntry ]={}
        self .expertise_embeddings :Dict [str ,np .ndarray ]={}
        logger .info ("AEP Repository initialized")

    def register_agent (self ,entry :AEPEntry )->bool :
        try :
            self .entries [entry .agent_id ]=entry 
            logger .info (f"Agent registered in AEP: {entry .agent_id } - {entry .specialty }")
            return True 
        except Exception as e :
            logger .error (f"Failed to register agent {entry .agent_id }: {e }")
            return False 

    def get_agent_entry (self ,agent_id :str )->Optional [AEPEntry ]:
        return self .entries .get (agent_id )

    def update_performance_history (self ,agent_id :str ,task_type :str ,performance_score :float ):
        if agent_id in self .entries :
            self .entries [agent_id ].performance_history [task_type ]=performance_score 
            self .entries [agent_id ].last_updated =datetime .now ().isoformat ()

    def get_all_entries (self )->Dict [str ,AEPEntry ]:
        return self .entries .copy ()

    def find_agents_by_capability (self ,capability :str )->List [AEPEntry ]:
        matching_agents =[]
        for entry in self .entries .values ():
            if capability .lower ()in [cap .lower ()for cap in entry .capabilities ]:
                matching_agents .append (entry )
        return matching_agents 

    def export_to_json (self ,filepath :str ):
        try :
            export_data ={
            agent_id :entry .to_dict ()
            for agent_id ,entry in self .entries .items ()
            }
            with open (filepath ,'w')as f :
                json .dump (export_data ,f ,indent =2 )
            logger .info (f"AEP repository exported to {filepath }")
        except Exception as e :
            logger .error (f"Failed to export AEP repository: {e }")

_aep_repository =None 

def get_aep_repository ()->AEPRepository :
    global _aep_repository 
    if _aep_repository is None :
        _aep_repository =AEPRepository ()
        _register_default_agents (_aep_repository )
    return _aep_repository 

def _register_default_agents (repo :AEPRepository ):
    default_agents =[
    AEPEntry (
    agent_id ="weather_agent",
    agent_name ="WeatherAgent",
    specialty ="Meteorological Analysis and Weather Impact Assessment",
    output_type ="Weather Score [0-1] and Detailed Analysis",
    capabilities =["weather_analysis","meteorological_assessment","flight_weather_impact"],
    expertise_areas =["aviation_weather","meteorology","weather_hazards","visibility_analysis"]
    ),
    AEPEntry (
    agent_id ="safety_assessment_agent",
    agent_name ="SafetyAssessmentAgent",
    specialty ="Aviation Safety Analysis and Risk Assessment",
    output_type ="Safety Score [0-1] and Risk Analysis",
    capabilities =["safety_analysis","risk_assessment","icao_compliance","accident_analysis"],
    expertise_areas =["aviation_safety","flight_safety","aircraft_safety","airport_safety"]
    ),
    AEPEntry (
    agent_id ="flight_info_agent",
    agent_name ="FlightInfoAgent",
    specialty ="Flight Operations and Schedule Analysis",
    output_type ="Operational Score [0-1] and Flight Details",
    capabilities =["flight_analysis","schedule_optimization","operational_assessment"],
    expertise_areas =["flight_operations","airline_operations","flight_scheduling","aircraft_performance"]
    ),
    AEPEntry (
    agent_id ="economic_agent",
    agent_name ="EconomicAgent",
    specialty ="Economic Analysis and Cost Optimization",
    output_type ="Economic Score [0-1] and Cost Analysis",
    capabilities =["cost_analysis","economic_assessment","price_optimization","budget_analysis"],
    expertise_areas =["aviation_economics","flight_costs","airline_pricing","travel_economics"]
    ),
    AEPEntry (
    agent_id ="integration_agent",
    agent_name ="IntegrationAgent",
    specialty ="Multi-Agent Result Integration and Decision Synthesis",
    output_type ="Integrated Score [0-1] and Final Recommendations",
    capabilities =["result_integration","decision_synthesis","multi_criteria_analysis"],
    expertise_areas =["decision_integration","multi_agent_coordination","result_synthesis"]
    )
    ]

    for agent in default_agents :
        repo .register_agent (agent )

@dataclass 
class QueryVector :

    vector :np .ndarray 
    query_text :str 
    timestamp :str 
    model_name :str 

@dataclass 
class ComputationResult :

    similarity_scores :np .ndarray 
    query_vector :QueryVector 
    computation_time :float 
    agent_matches :List [Tuple [str ,float ]]

EMBEDDING_CACHE ={}
CACHE_DIR =Path (config .cache_dir )/"embeddings"

def _get_text_hash (text :str )->str :

    return hashlib .md5 (text .encode ()).hexdigest ()

class SBERTSimilarityEngine :
    def __init__ (self ,model_name :str =None ,cache_dir :str =None ):
        self .model_name =model_name or config .sbert_model_name 
        self .cache_dir =Path (cache_dir or CACHE_DIR )
        self .cache_dir .mkdir (parents =True ,exist_ok =True )

        try :
            model_cache_dir =self .cache_dir /"models"
            model_cache_dir .mkdir (parents =True ,exist_ok =True )

            self .model =SentenceTransformer (self .model_name ,cache_folder =str (model_cache_dir ))
            self .embedding_dim =self .model .get_sentence_embedding_dimension ()
            logger .info (f"SBERT model loaded with caching: {self .model_name } (dim: {self .embedding_dim })")
        except Exception as e :
            class _FallbackEncoder :
                def __init__ (self ,dim :int =config .embedding_dimension ):
                    self ._dim =dim 
                def get_sentence_embedding_dimension (self ):
                    return self ._dim 
                def encode (self ,text :str ,convert_to_numpy :bool =True ):
                    h =hashlib .md5 (str (text ).encode ()).digest ()
                    base =np .frombuffer (h ,dtype =np .uint8 ).astype (np .float32 )
                    repeats =int (np .ceil (config .embedding_dimension /len (base )))
                    vec =np .tile (base ,repeats )[:config .embedding_dimension ]
                    vec =vec /max (1.0 ,np .linalg .norm (vec ))
                    return vec 
            self .model =_FallbackEncoder ()
            self .embedding_dim =self .model .get_sentence_embedding_dimension ()
            logger .warning (f"Using fallback SBERT encoder: {self .model_name } unavailable")

        self .agent_embeddings :Dict [str ,np .ndarray ]={}
        self .agent_expertise :Dict [str ,Dict [str ,Any ]]={}

        self .computation_stats ={
        'total_queries':0 ,
        'cache_hits':0 ,
        'cache_misses':0 ,
        'average_computation_time':0.0 
        }

        self ._load_cached_embeddings ()

    def encode_agent_expertise (self ,agent_id :str ,expertise_texts :List [str ],
    expertise_area :str ,capabilities :List [str ])->bool :
        try :
            combined_text =" ".join (expertise_texts )

            start_time =time .time ()
            embedding =self .model .encode (combined_text ,convert_to_numpy =True )
            computation_time =time .time ()-start_time 

            self .agent_embeddings [agent_id ]=embedding 
            self .agent_expertise [agent_id ]={
            'expertise_texts':expertise_texts ,
            'expertise_area':expertise_area ,
            'capabilities':capabilities ,
            'combined_text':combined_text ,
            'embedding_timestamp':datetime .now ().isoformat (),
            'computation_time':computation_time 
            }

            self ._cache_embedding (f"agent_{agent_id }",combined_text ,embedding )

            logger .debug (f"Encoded expertise for agent {agent_id } in {computation_time :.3f}s")
            return True 

        except Exception as e :
            logger .error (f"Failed to encode expertise for agent {agent_id }: {e }")
            return False 

    def compute_similarity (self ,text1 :str ,text2 :str )->float :
        try :
            embedding1 =self ._get_query_embedding (text1 )
            embedding2 =self ._get_query_embedding (text2 )

            similarity =np .dot (embedding1 ,embedding2 )/(
            np .linalg .norm (embedding1 )*np .linalg .norm (embedding2 )
            )

            return max (0.0 ,min (1.0 ,similarity ))

        except Exception as e :
            logger .error (f"Failed to compute similarity: {e }")
            return 0.0 

    def compute_similarity_with_agent (self ,query_text :str ,agent_id :str )->float :
        try :
            if agent_id not in self .agent_embeddings :
                logger .warning (f"No embedding found for agent {agent_id }")
                return 0.0 

            query_embedding =self ._get_query_embedding (query_text )
            agent_embedding =self .agent_embeddings [agent_id ]

            similarity =cosine_similarity (
            query_embedding .reshape (1 ,-1 ),
            agent_embedding .reshape (1 ,-1 )
            )[0 ,0 ]

            similarity =max (0.0 ,min (1.0 ,similarity ))

            self .computation_stats ['total_queries']+=1 

            return float (similarity )

        except Exception as e :
            logger .error (f"Failed to compute similarity with agent {agent_id }: {e }")
            return 0.0 

    def compute_similarities_batch (self ,query_text :str ,agent_ids :List [str ]=None )->Dict [str ,float ]:
        try :
            start_time =time .time ()

            if agent_ids is None :
                agent_ids =list (self .agent_embeddings .keys ())

            query_embedding =self ._get_query_embedding (query_text )

            similarities ={}
            for agent_id in agent_ids :
                if agent_id in self .agent_embeddings :
                    agent_embedding =self .agent_embeddings [agent_id ]
                    similarity =cosine_similarity (
                    query_embedding .reshape (1 ,-1 ),
                    agent_embedding .reshape (1 ,-1 )
                    )[0 ,0 ]
                    similarities [agent_id ]=max (0.0 ,min (1.0 ,float (similarity )))
                else :
                    similarities [agent_id ]=0.0 

            computation_time =time .time ()-start_time 
            self .computation_stats ['total_queries']+=1 

            self .computation_stats ['average_computation_time']=(
            (self .computation_stats ['average_computation_time']*(self .computation_stats ['total_queries']-1 )+
            computation_time )/self .computation_stats ['total_queries']
            )

            logger .debug (f"Computed similarities for {len (agent_ids )} agents in {computation_time :.3f}s")
            return similarities 

        except Exception as e :
            logger .error (f"Failed to compute batch similarities: {e }")
            return {agent_id :0.0 for agent_id in (agent_ids or [])}

    def _get_query_embedding (self ,query_text :str )->np .ndarray :
        query_hash =_get_text_hash (query_text )

        if config .enable_embedding_cache and query_hash in EMBEDDING_CACHE :
            self .computation_stats ['cache_hits']+=1 
            return EMBEDDING_CACHE [query_hash ]['embedding']

        start_time =time .time ()
        embedding =self .model .encode (query_text ,convert_to_numpy =True )
        computation_time =time .time ()-start_time 

        if config .enable_embedding_cache :
            EMBEDDING_CACHE [query_hash ]={
            'embedding':embedding ,
            'text':query_text ,
            'timestamp':datetime .now ().isoformat (),
            'computation_time':computation_time 
            }
            self ._cache_embedding (f"query_{query_hash }",query_text ,embedding )

        self .computation_stats ['cache_misses']+=1 
        return embedding 

    def _cache_embedding (self ,key :str ,text :str ,embedding :np .ndarray ):
        try :
            if config .enable_embedding_cache :
                cache_file =self .cache_dir /f"{key }.pkl"
                cache_data ={
                'text':text ,
                'embedding':embedding ,
                'timestamp':datetime .now ().isoformat (),
                'model_name':self .model_name 
                }
                with open (cache_file ,'wb')as f :
                    pickle .dump (cache_data ,f )
        except Exception as e :
            logger .warning (f"Failed to cache embedding for {key }: {e }")

    def _load_cached_embeddings (self ):
        try :
            if not config .enable_embedding_cache :
                return 

            cache_files =list (self .cache_dir .glob ("*.pkl"))
            loaded_count =0 

            for cache_file in cache_files :
                try :
                    with open (cache_file ,'rb')as f :
                        cache_data =pickle .load (f )

                    cache_time =datetime .fromisoformat (cache_data ['timestamp'])
                    if (datetime .now ()-cache_time ).total_seconds ()<config .cache_expiry_hours *3600 :
                        key =cache_file .stem 
                        if key .startswith ('query_'):
                            query_hash =key .replace ('query_','')
                            EMBEDDING_CACHE [query_hash ]=cache_data 
                        elif key .startswith ('agent_'):
                            agent_id =key .replace ('agent_','')
                            self .agent_embeddings [agent_id ]=cache_data ['embedding']

                        loaded_count +=1 

                except Exception as e :
                    logger .warning (f"Failed to load cached embedding {cache_file }: {e }")

            if loaded_count >0 :
                logger .info (f"Loaded {loaded_count } cached embeddings")

        except Exception as e :
            logger .warning (f"Failed to load cached embeddings: {e }")

    def get_statistics (self )->Dict [str ,Any ]:
        return {
        'model_name':self .model_name ,
        'embedding_dimension':self .embedding_dim ,
        'agents_encoded':len (self .agent_embeddings ),
        'computation_stats':self .computation_stats .copy (),
        'cache_hit_rate':(
        self .computation_stats ['cache_hits']/
        max (1 ,self .computation_stats ['cache_hits']+self .computation_stats ['cache_misses'])
        )
        }

_sbert_engine =None 

def get_sbert_engine ()->SBERTSimilarityEngine :

    global _sbert_engine 
    if _sbert_engine is None :
        _sbert_engine =SBERTSimilarityEngine ()
    return _sbert_engine 

class ActionType (Enum ):
    SELECT_AGENT ="select_agent"
    COORDINATE_AGENTS ="coordinate_agents"
    TRUST_UPDATE ="trust_update"
    PERFORMANCE_EVALUATION ="performance_evaluation"

class StateType (Enum ):
    QUERY_STATE ="query_state"
    AGENT_STATE ="agent_state"
    SYSTEM_STATE ="system_state"
    TRUST_STATE ="trust_state"

@dataclass 
class MARLState :
    query_text :str 
    available_agents :List [str ]
    context :Dict [str ,Any ]
    timestamp :datetime =field (default_factory =datetime .now )
    state_id :str =field (default_factory =lambda :str (uuid .uuid4 ()))

@dataclass 
class MARLAction :
    action_type :ActionType 
    agent_id :str 
    parameters :Dict [str ,Any ]
    timestamp :datetime =field (default_factory =datetime .now )
    action_id :str =field (default_factory =lambda :str (uuid .uuid4 ()))

@dataclass 
class MARLReward :
    reward_value :float 
    reward_components :Dict [str ,float ]
    timestamp :datetime =field (default_factory =datetime .now )

@dataclass 
class AgentQTable :
    agent_id :str 
    q_values :Dict [str ,Dict [str ,float ]]
    learning_rate :float 
    discount_factor :float 
    epsilon :float 
    last_updated :datetime =field (default_factory =datetime .now )

class TrustAwareMARLEngine :
    def __init__ (self ,learning_rate :float =None ,discount_factor :float =None ,
    trust_weight :float =None ,config :Dict [str ,Any ]=None ):
        self .config =config or {}
        global_config =get_config ()

        self .learning_rate =learning_rate or getattr (global_config ,'learning_rate',0.001 )
        self .discount_factor =discount_factor or getattr (global_config ,'discount_factor',0.95 )
        self .trust_weight =trust_weight or getattr (global_config ,'trust_weight',0.3 )

        self .epsilon =getattr (global_config ,'epsilon_start',1.0 )
        self .epsilon_min =getattr (global_config ,'epsilon_end',0.01 )
        self .epsilon_decay =getattr (global_config ,'epsilon_decay',0.995 )

        self .agent_q_tables :Dict [str ,AgentQTable ]={}

        self .state_dimension =getattr (global_config ,'state_dimension',10 )
        self .action_dimension =getattr (global_config ,'action_dimension',5 )

        self .experience_buffer =deque (maxlen =10000 )

        self .performance_metrics ={
        'total_episodes':0 ,
        'total_rewards':0.0 ,
        'average_reward':0.0 ,
        'convergence_history':[],
        'trust_alignment_history':[]
        }

        self .trust_scores :Dict [str ,float ]={}

        logger .info (f"MARL Engine initialized: lr={self .learning_rate }, gamma={self .discount_factor }, trust_weight={self .trust_weight }")

    def create_state (self ,query_text :str ,available_agents :List [str ],
    system_context :Dict [str ,Any ])->MARLState :
        return MARLState (
        query_text =query_text ,
        available_agents =available_agents ,
        context =system_context 
        )

    def select_agents (self ,state :MARLState ,num_agents :int ,
    selection_strategy :str ="trust_weighted_semantic")->List [Tuple [str ,float ]]:
        try :
            available_agents =state .available_agents 

            if not available_agents :
                return []

            semantic_similarities =state .context .get ('semantic_similarities',{})

            agent_scores =[]

            for agent_id in available_agents :
                q_value =self ._get_q_value (state ,agent_id )
                trust_score =self .trust_scores .get (agent_id ,0.5 )
                semantic_score =semantic_similarities .get (agent_id ,0.5 )

                if selection_strategy =="trust_weighted_semantic":

                    selection_score =(
                    config .selection_alpha *semantic_score +
                    (1 -config .selection_alpha )*trust_score +
                    0.1 *q_value 
                    )
                elif selection_strategy =="pure_trust":
                    selection_score =trust_score 
                elif selection_strategy =="pure_semantic":
                    selection_score =semantic_score 
                else :
                    selection_score =0.4 *semantic_score +0.4 *trust_score +0.2 *q_value 

                agent_scores .append ((agent_id ,selection_score ))

            agent_scores .sort (key =lambda x :x [1 ],reverse =True )
            selected_agents =agent_scores [:num_agents ]

            logger .debug (f"Selected {len (selected_agents )} agents using {selection_strategy } strategy")
            return selected_agents 

        except Exception as e :
            logger .error (f"Agent selection failed: {e }")
            return [(agent_id ,0.5 )for agent_id in available_agents [:num_agents ]]

    def calculate_reward (self ,mrr :float ,ndcg_at_5 :float ,art :float ,
    lambda1 :float =0.4 ,lambda2 :float =0.4 ,lambda3 :float =0.2 )->float :
        try :
            normalized_art =min (art /30.0 ,1.0 )if art >0 else 0.0 

            reward =lambda1 *mrr +lambda2 *ndcg_at_5 -lambda3 *normalized_art 

            reward =max (-1.0 ,min (1.0 ,reward ))

            return reward 

        except Exception as e :
            logger .error (f"Reward calculation failed: {e }")
            return 0.0 

    def coordinate_agents (self ,selected_agents :List [str ],
    coordination_strategy :str ="collaborative")->Dict [str ,Any ]:
        try :
            coordination_result ={
            'strategy':coordination_strategy ,
            'agents':selected_agents ,
            'coordination_quality':0.0 ,
            'trust_consistency':0.0 ,
            'q_learning_updates':0 ,
            'byzantine_tolerance':True 
            }

            if coordination_strategy =="collaborative":
                trust_scores =[self .trust_scores .get (agent_id ,0.5 )for agent_id in selected_agents ]
                coordination_result ['coordination_quality']=np .mean (trust_scores )
                coordination_result ['trust_consistency']=1.0 -np .std (trust_scores )

            elif coordination_strategy =="competitive":
                agent_performances =[trust_scores .get (agent ,0.5 )for agent in selected_agents ]
                coordination_result ['coordination_quality']=max (agent_performances )if agent_performances else 0.7 
                coordination_result ['trust_consistency']=1.0 -(max (agent_performances )-min (agent_performances ))if len (agent_performances )>1 else 0.6 

            self ._update_coordination_q_values (selected_agents ,coordination_result )
            coordination_result ['q_learning_updates']=len (selected_agents )

            return coordination_result 

        except Exception as e :
            logger .error (f"Agent coordination failed: {e }")
            return {
            'strategy':coordination_strategy ,
            'agents':selected_agents ,
            'coordination_quality':0.0 ,
            'error':str (e )
            }

    def _get_q_value (self ,state :MARLState ,agent_id :str )->float :
        try :
            if agent_id not in self .agent_q_tables :
                self ._initialize_agent_q_table (agent_id )

            q_table =self .agent_q_tables [agent_id ]
            state_key =self ._state_to_key (state )

            if state_key in q_table .q_values :
                return q_table .q_values [state_key ].get ('select',0.0 )
            else :
                return 0.0 

        except Exception as e :
            logger .warning (f"Failed to get Q-value for {agent_id }: {e }")
            return 0.0 

    def _initialize_agent_q_table (self ,agent_id :str ):
        self .agent_q_tables [agent_id ]=AgentQTable (
        agent_id =agent_id ,
        q_values ={},
        learning_rate =self .learning_rate ,
        discount_factor =self .discount_factor ,
        epsilon =self .epsilon 
        )

    def _state_to_key (self ,state :MARLState )->str :
        context_hash =hashlib .md5 (str (sorted (state .context .items ())).encode ()).hexdigest ()[:8 ]
        return f"query_{len (state .query_text )}_{len (state .available_agents )}_{context_hash }"

    def _update_coordination_q_values (self ,selected_agents :List [str ],coordination_result :Dict [str ,Any ]):
        try :
            reward =coordination_result .get ('coordination_quality',0.0 )

            for agent_id in selected_agents :
                if agent_id in self .agent_q_tables :
                    q_table =self .agent_q_tables [agent_id ]
                    state_key =f"coordination_{len (selected_agents )}"
                    if state_key not in q_table .q_values :
                        q_table .q_values [state_key ]={}

                    current_q =q_table .q_values [state_key ].get ('coordinate',0.0 )
                    new_q =current_q +self .learning_rate *(reward -current_q )
                    q_table .q_values [state_key ]['coordinate']=new_q 
                    q_table .last_updated =datetime .now ()

        except Exception as e :
            logger .warning (f"Failed to update coordination Q-values: {e }")

    def update_trust_scores (self ,trust_scores :Dict [str ,float ]):
        self .trust_scores .update (trust_scores )
        logger .debug (f"Updated trust scores for {len (trust_scores )} agents")

    def update_q_values_with_reward (self ,state :MARLState ,selected_agents :List [str ],
    reward :float ,next_state :MARLState =None ):
        try :
            state_key =self ._state_to_key (state )

            for agent_id in selected_agents :
                if agent_id not in self .agent_q_tables :
                    self ._initialize_agent_q_table (agent_id )

                q_table =self .agent_q_tables [agent_id ]

                if state_key not in q_table .q_values :
                    q_table .q_values [state_key ]={}

                current_q =q_table .q_values [state_key ].get ('select',0.0 )

                next_q_max =0.0 
                if next_state :
                    next_state_key =self ._state_to_key (next_state )
                    if next_state_key in q_table .q_values :
                        next_q_max =max (q_table .q_values [next_state_key ].values ())

                td_target =reward +self .discount_factor *next_q_max 
                td_error =td_target -current_q 
                new_q =current_q +self .learning_rate *td_error 
                q_table .q_values [state_key ]['select']=new_q 
                q_table .last_updated =datetime .now ()
                self .performance_metrics ['total_episodes']+=1 
                self .performance_metrics ['total_rewards']+=reward 
                self .performance_metrics ['average_reward']=(
                self .performance_metrics ['total_rewards']/
                self .performance_metrics ['total_episodes']
                )

                self .performance_metrics ['convergence_history'].append (abs (td_error ))
                if len (self .performance_metrics ['convergence_history'])>1000 :
                    self .performance_metrics ['convergence_history']=(
                    self .performance_metrics ['convergence_history'][-1000 :]
                    )

            self .epsilon =max (self .epsilon_min ,self .epsilon *self .epsilon_decay )

            logger .debug (f"Updated Q-values for {len (selected_agents )} agents with reward {reward :.4f}")

        except Exception as e :
            logger .error (f"Q-value update with reward failed: {e }")

    def batch_update_from_experience (self ,experiences :List [Dict [str ,Any ]]):
        try :
            if not experiences :
                return 

            logger .info (f"Performing batch Q-value update with {len (experiences )} experiences")

            for experience in experiences :

                agent_outputs =experience .get ('agent_outputs',{})
                trust_scores =experience .get ('trust_scores',{})
                reward =experience .get ('reward',0.0 )

                available_agents =list (agent_outputs .keys ())
                state =MARLState (
                query_text =f"batch_update_{experience .get ('query_id','unknown')}",
                available_agents =available_agents ,
                context ={'trust_scores':trust_scores }
                )

                self .update_q_values_with_reward (state ,available_agents ,reward )

            logger .info (f"Batch Q-value update completed for {len (experiences )} experiences")

        except Exception as e :
            logger .error (f"Batch Q-value update failed: {e }")

    def get_performance_metrics (self )->Dict [str ,Any ]:
        return self .performance_metrics .copy ()

    def save_model (self ,filepath :str ):
        try :
            model_data ={
            'agent_q_tables':{
            agent_id :{
            'agent_id':q_table .agent_id ,
            'q_values':q_table .q_values ,
            'learning_rate':q_table .learning_rate ,
            'discount_factor':q_table .discount_factor ,
            'epsilon':q_table .epsilon ,
            'last_updated':q_table .last_updated .isoformat ()
            }
            for agent_id ,q_table in self .agent_q_tables .items ()
            },
            'performance_metrics':self .performance_metrics ,
            'trust_scores':self .trust_scores ,
            'config':{
            'learning_rate':self .learning_rate ,
            'discount_factor':self .discount_factor ,
            'trust_weight':self .trust_weight 
            }
            }

            with open (filepath ,'w')as f :
                json .dump (model_data ,f ,indent =2 )

            logger .info (f"MARL model saved to {filepath }")

        except Exception as e :
            logger .error (f"Failed to save MARL model: {e }")

    def load_model (self ,filepath :str ):
        try :
            with open (filepath ,'r')as f :
                model_data =json .load (f )

            self .agent_q_tables ={}
            for agent_id ,q_data in model_data .get ('agent_q_tables',{}).items ():
                self .agent_q_tables [agent_id ]=AgentQTable (
                agent_id =q_data ['agent_id'],
                q_values =q_data ['q_values'],
                learning_rate =q_data ['learning_rate'],
                discount_factor =q_data ['discount_factor'],
                epsilon =q_data ['epsilon'],
                last_updated =datetime .fromisoformat (q_data ['last_updated'])
                )

            self .performance_metrics =model_data .get ('performance_metrics',{})
            self .trust_scores =model_data .get ('trust_scores',{})

            logger .info (f"MARL model loaded from {filepath }")

        except Exception as e :
            logger .error (f"Failed to load MARL model: {e }")

_marl_engine =None 

def get_marl_engine ()->TrustAwareMARLEngine :

    global _marl_engine 
    if _marl_engine is None :
        _marl_engine =TrustAwareMARLEngine (
        learning_rate =config .learning_rate ,
        discount_factor =config .discount_factor ,
        trust_weight =config .trust_weight ,
        config =config 
        )
    return _marl_engine 

class RankingAlgorithm (Enum ):
    POINTWISE ="pointwise"
    PAIRWISE ="pairwise"
    LISTWISE ="listwise"
    RANKNET ="ranknet"
    LAMBDARANK ="lambdarank"
    LISTNET ="listnet"

@dataclass 
class RankingQuery :
    query_id :str 
    query_text :str 
    features :np .ndarray 
    preferences :Dict [str ,Any ]
    timestamp :datetime =field (default_factory =datetime .now )

@dataclass 
class RankingInstance :
    instance_id :str 
    features :np .ndarray 
    relevance_score :float 
    metadata :Dict [str ,Any ]

class LTRNeuralNetwork (nn .Module ):
    def __init__ (self ,input_dim :int ,hidden_dims :List [int ],output_dim :int =1 ):
        super (LTRNeuralNetwork ,self ).__init__ ()

        layers =[]
        prev_dim =input_dim 

        for hidden_dim in hidden_dims :
            layers .extend ([
            nn .Linear (prev_dim ,hidden_dim ),
            nn .ReLU (),
            nn .Dropout (0.2 )
            ])
            prev_dim =hidden_dim 

        layers .append (nn .Linear (prev_dim ,output_dim ))

        self .network =nn .Sequential (*layers )

    def forward (self ,x ):
        return self .network (x )

class LTRRankingEngine :
    def __init__ (self ,algorithm :RankingAlgorithm =RankingAlgorithm .LISTWISE ,
    feature_dim :int =None ,hidden_dims :List [int ]=None ,
    learning_rate :float =None ,batch_size :int =None ,
    num_epochs :int =None ,device :str ="cpu"):
        self .algorithm =algorithm 
        self .feature_dim =feature_dim or config .feature_dimension 
        self .hidden_dims =hidden_dims or config .hidden_dimensions 
        self .learning_rate =learning_rate or config .ltr_learning_rate 
        self .batch_size =batch_size or config .batch_size 
        self .num_epochs =num_epochs or config .num_epochs 
        self .device =device 

        self .model =LTRNeuralNetwork (
        input_dim =self .feature_dim ,
        hidden_dims =self .hidden_dims ,
        output_dim =1 
        ).to (self .device )

        self .optimizer =optim .Adam (self .model .parameters (),lr =self .learning_rate )
        self .criterion =self ._get_loss_function ()

        self .scaler =StandardScaler ()
        self .scaler_fitted =False 

        self .training_history ={
        'losses':[],
        'ndcg_scores':[],
        'epochs_trained':0 
        }

        self .performance_metrics ={
        'total_queries_ranked':0 ,
        'average_ranking_time':0.0 ,
        'average_ndcg':0.0 ,
        'model_accuracy':0.0 
        }

        logger .info (f"LTR Engine initialized: {algorithm .value }, feature_dim={self .feature_dim }")

    def _get_loss_function (self ):
        if self .algorithm ==RankingAlgorithm .POINTWISE :
            return nn .MSELoss ()
        elif self .algorithm ==RankingAlgorithm .PAIRWISE :
            return nn .BCEWithLogitsLoss ()
        elif self .algorithm ==RankingAlgorithm .LISTWISE :
            return self ._listwise_loss 
        elif self .algorithm ==RankingAlgorithm .RANKNET :
            return self ._ranknet_loss 
        elif self .algorithm ==RankingAlgorithm .LAMBDARANK :
            return self ._lambdarank_loss 
        else :
            return nn .MSELoss ()

    def _listwise_loss (self ,predictions ,targets ):
        try :
            pred_probs =torch .softmax (predictions ,dim =0 )
            target_probs =torch .softmax (targets ,dim =0 )

            loss =-torch .sum (target_probs *torch .log (pred_probs +1e-10 ))
            return loss 
        except Exception as e :
            logger .warning (f"Listwise loss computation failed: {e }")
            return torch .tensor (0.0 ,requires_grad =True )

    def _ranknet_loss (self ,predictions ,targets ):
        try :
            if len (predictions )<2 :
                return torch .tensor (0.0 ,requires_grad =True )

            total_loss =torch .tensor (0.0 ,requires_grad =True ,device =predictions .device )
            pair_count =0 

            for i in range (len (predictions )):
                for j in range (len (predictions )):
                    if i !=j :
                        score_diff =predictions [i ]-predictions [j ]

                        if targets [i ]>targets [j ]:
                            true_preference =torch .tensor (1.0 ,device =predictions .device )
                        elif targets [i ]<targets [j ]:
                            true_preference =torch .tensor (0.0 ,device =predictions .device )
                        else :
                            continue 

                        pred_preference =torch .sigmoid (score_diff )

                        pair_loss =-(true_preference *torch .log (pred_preference +1e-10 )+
                        (1 -true_preference )*torch .log (1 -pred_preference +1e-10 ))

                        total_loss =total_loss +pair_loss 
                        pair_count +=1 

            if pair_count >0 :
                return total_loss /pair_count 
            else :
                return torch .tensor (0.0 ,requires_grad =True )

        except Exception as e :
            logger .warning (f"RankNet loss computation failed: {e }")
            return torch .tensor (0.0 ,requires_grad =True )

    def _lambdarank_loss (self ,predictions ,targets ):
        try :
            if len (predictions )<2 :
                return torch .tensor (0.0 ,requires_grad =True )

            pred_np =predictions .detach ().cpu ().numpy ()
            targets_np =targets .detach ().cpu ().numpy ()

            current_ndcg =self ._calculate_ndcg (targets_np ,pred_np )

            total_loss =torch .tensor (0.0 ,requires_grad =True ,device =predictions .device )
            pair_count =0 

            for i in range (len (predictions )):
                for j in range (len (predictions )):
                    if i !=j and targets [i ]!=targets [j ]:
                        score_diff =predictions [i ]-predictions [j ]

                        swapped_pred =pred_np .copy ()
                        swapped_pred [i ],swapped_pred [j ]=swapped_pred [j ],swapped_pred [i ]
                        swapped_ndcg =self ._calculate_ndcg (targets_np ,swapped_pred )

                        ndcg_change =abs (swapped_ndcg -current_ndcg )

                        sigmoid_val =torch .sigmoid (score_diff )
                        sigmoid_derivative =sigmoid_val *(1 -sigmoid_val )

                        lambda_ij =sigmoid_derivative *ndcg_change 

                        if targets [i ]>targets [j ]:
                            true_preference =torch .tensor (1.0 ,device =predictions .device )
                        else :
                            true_preference =torch .tensor (0.0 ,device =predictions .device )

                        pred_preference =torch .sigmoid (score_diff )

                        pair_loss =lambda_ij *(-(true_preference *torch .log (pred_preference +1e-10 )+
                        (1 -true_preference )*torch .log (1 -pred_preference +1e-10 )))

                        total_loss =total_loss +pair_loss 
                        pair_count +=1 

            if pair_count >0 :
                return total_loss /pair_count 
            else :
                return torch .tensor (0.0 ,requires_grad =True )

        except Exception as e :
            logger .warning (f"LambdaRank loss computation failed: {e }")
            return torch .tensor (0.0 ,requires_grad =True )

    def _compute_pairwise_loss (self ,predictions ,targets ):
        try :
            if len (predictions )<2 :
                return torch .tensor (0.0 ,requires_grad =True )

            total_loss =torch .tensor (0.0 ,requires_grad =True ,device =predictions .device )
            pair_count =0 

            for i in range (len (predictions )):
                for j in range (len (predictions )):
                    if i !=j and targets [i ]!=targets [j ]:
                        if targets [i ]>targets [j ]:
                            pairwise_label =torch .tensor (1.0 ,device =predictions .device )
                        else :
                            pairwise_label =torch .tensor (0.0 ,device =predictions .device )

                        score_diff =predictions [i ]-predictions [j ]

                        pred_prob =torch .sigmoid (score_diff )
                        pair_loss =-(pairwise_label *torch .log (pred_prob +1e-10 )+
                        (1 -pairwise_label )*torch .log (1 -pred_prob +1e-10 ))

                        total_loss =total_loss +pair_loss 
                        pair_count +=1 

            if pair_count >0 :
                return total_loss /pair_count 
            else :
                return torch .tensor (0.0 ,requires_grad =True )

        except Exception as e :
            logger .warning (f"Pairwise loss computation failed: {e }")
            return torch .tensor (0.0 ,requires_grad =True )

    def rank_items (self ,query :RankingQuery ,items :List [RankingInstance ])->List [Tuple [str ,float ,Dict [str ,Any ]]]:
        try :
            start_time =time .time ()

            if not items :
                return []

            item_features =[]
            item_ids =[]
            item_metadata =[]

            for item in items :
                combined_features =self ._combine_features (query .features ,item .features )
                item_features .append (combined_features )
                item_ids .append (item .instance_id )
                item_metadata .append (item .metadata )

            features_tensor =torch .FloatTensor (np .array (item_features )).to (self .device )

            self .model .eval ()
            with torch .no_grad ():
                predictions =self .model (features_tensor ).squeeze ()

            if predictions .dim ()==0 :
                predictions =predictions .unsqueeze (0 )

            scores =predictions .cpu ().numpy ()

            ranked_results =[]
            for i ,(item_id ,score ,metadata )in enumerate (zip (item_ids ,scores ,item_metadata )):
                ranked_results .append ((item_id ,float (score ),metadata ))

            ranked_results .sort (key =lambda x :x [1 ],reverse =True )

            ranking_time =time .time ()-start_time 
            self .performance_metrics ['total_queries_ranked']+=1 
            self .performance_metrics ['average_ranking_time']=(
            (self .performance_metrics ['average_ranking_time']*(self .performance_metrics ['total_queries_ranked']-1 )+
            ranking_time )/self .performance_metrics ['total_queries_ranked']
            )

            logger .debug (f"Ranked {len (items )} items in {ranking_time :.3f}s")
            return ranked_results 

        except Exception as e :
            logger .error (f"Ranking failed: {e }")
            return [(item .instance_id ,0.5 ,item .metadata )for item in items ]

    def _combine_features (self ,query_features :np .ndarray ,item_features :np .ndarray )->np .ndarray :
        try :
            if len (query_features )!=len (item_features ):
                query_features =self ._normalize_features (query_features )
                item_features =self ._normalize_features (item_features )

            combined =np .concatenate ([
            query_features ,
            item_features ,
            query_features *item_features ,
            np .abs (query_features -item_features )
            ])

            if len (combined )>self .feature_dim :
                combined =combined [:self .feature_dim ]
            elif len (combined )<self .feature_dim :
                padding =np .zeros (self .feature_dim -len (combined ))
                combined =np .concatenate ([combined ,padding ])

            return combined 

        except Exception as e :
            logger .warning (f"Feature combination failed: {e }")
            return np .zeros (self .feature_dim )

    def _normalize_features (self ,features :np .ndarray )->np .ndarray :
        target_dim =self .feature_dim //4 

        if len (features )>target_dim :
            return features [:target_dim ]
        elif len (features )<target_dim :
            padding =np .zeros (target_dim -len (features ))
            return np .concatenate ([features ,padding ])
        else :
            return features 

    def train (self ,training_data :List [Tuple [RankingQuery ,List [RankingInstance ],List [float ]]]):
        try :
            if not training_data :
                logger .warning ("No training data provided")
                return 

            logger .info (f"Training LTR model with {len (training_data )} queries")

            all_features =[]
            all_targets =[]

            for query ,items ,relevance_scores in training_data :
                for item ,relevance in zip (items ,relevance_scores ):
                    combined_features =self ._combine_features (query .features ,item .features )
                    all_features .append (combined_features )
                    all_targets .append (relevance )

            if not all_features :
                logger .warning ("No valid training features generated")
                return 

            features_tensor =torch .FloatTensor (np .array (all_features )).to (self .device )
            targets_tensor =torch .FloatTensor (all_targets ).to (self .device )

            if not self .scaler_fitted :
                self .scaler .fit (all_features )
                self .scaler_fitted =True 

            self .model .train ()
            for epoch in range (self .num_epochs ):
                self .optimizer .zero_grad ()

                predictions =self .model (features_tensor ).squeeze ()

                if self .algorithm ==RankingAlgorithm .LISTWISE :
                    loss =self .criterion (predictions ,targets_tensor )
                elif self .algorithm ==RankingAlgorithm .RANKNET :
                    loss =self .criterion (predictions ,targets_tensor )
                elif self .algorithm ==RankingAlgorithm .LAMBDARANK :
                    loss =self .criterion (predictions ,targets_tensor )
                elif self .algorithm ==RankingAlgorithm .PAIRWISE :
                    loss =self ._compute_pairwise_loss (predictions ,targets_tensor )
                else :
                    loss =self .criterion (predictions ,targets_tensor )

                loss .backward ()
                self .optimizer .step ()

                if epoch %50 ==0 :
                    self .training_history ['losses'].append (loss .item ())
                    logger .debug (f"Epoch {epoch }/{self .num_epochs }, Loss: {loss .item ():.4f}")

            self .training_history ['epochs_trained']+=self .num_epochs 
            logger .info (f"LTR model training completed: {self .num_epochs } epochs")

        except Exception as e :
            logger .error (f"LTR training failed: {e }")

    def evaluate (self ,test_data :List [Tuple [RankingQuery ,List [RankingInstance ],List [float ]]])->Dict [str ,float ]:
        try :
            if not test_data :
                return {'ndcg':0.0 ,'accuracy':0.0 }

            ndcg_scores =[]

            for query ,items ,true_relevance in test_data :
                ranked_results =self .rank_items (query ,items )

                if ranked_results :
                    predicted_scores =[score for _ ,score ,_ in ranked_results ]
                    ndcg =self ._calculate_ndcg (true_relevance ,predicted_scores )
                    ndcg_scores .append (ndcg )

            average_ndcg =np .mean (ndcg_scores )if ndcg_scores else 0.0 
            self .performance_metrics ['average_ndcg']=average_ndcg 

            return {
            'ndcg':average_ndcg ,
            'num_queries':len (test_data ),
            'ndcg_scores':ndcg_scores 
            }

        except Exception as e :
            logger .error (f"LTR evaluation failed: {e }")
            return {'ndcg':0.0 ,'accuracy':0.0 }

    def _calculate_ndcg (self ,true_relevance :List [float ],predicted_scores :List [float ],k :int =None )->float :
        try :
            k =k or config .ndcg_k 

            if len (true_relevance )!=len (predicted_scores ):
                return 0.0 

            sorted_indices =np .argsort (predicted_scores )[::-1 ]
            sorted_relevance =[true_relevance [i ]for i in sorted_indices ]

            dcg =0.0 
            for i ,rel in enumerate (sorted_relevance [:k ]):
                dcg +=(2 **rel -1 )/np .log2 (i +2 )

            ideal_relevance =sorted (true_relevance ,reverse =True )
            idcg =0.0 
            for i ,rel in enumerate (ideal_relevance [:k ]):
                idcg +=(2 **rel -1 )/np .log2 (i +2 )

            return dcg /idcg if idcg >0 else 0.0 

        except Exception as e :
            logger .warning (f"NDCG calculation failed: {e }")
            return 0.0 

    def save_model (self ,filepath :str ):
        try :
            model_state ={
            'model_state_dict':self .model .state_dict (),
            'optimizer_state_dict':self .optimizer .state_dict (),
            'scaler':self .scaler ,
            'scaler_fitted':self .scaler_fitted ,
            'training_history':self .training_history ,
            'performance_metrics':self .performance_metrics ,
            'config':{
            'algorithm':self .algorithm .value ,
            'feature_dim':self .feature_dim ,
            'hidden_dims':self .hidden_dims ,
            'learning_rate':self .learning_rate 
            }
            }

            torch .save (model_state ,filepath )
            logger .info (f"LTR model saved to {filepath }")

        except Exception as e :
            logger .error (f"Failed to save LTR model: {e }")

    def load_model (self ,filepath :str ):
        try :
            model_state =torch .load (filepath ,map_location =self .device )

            self .model .load_state_dict (model_state ['model_state_dict'])
            self .optimizer .load_state_dict (model_state ['optimizer_state_dict'])
            self .scaler =model_state ['scaler']
            self .scaler_fitted =model_state ['scaler_fitted']
            self .training_history =model_state ['training_history']
            self .performance_metrics =model_state ['performance_metrics']

            logger .info (f"LTR model loaded from {filepath }")

        except Exception as e :
            logger .error (f"Failed to load LTR model: {e }")

    def get_performance_metrics (self )->Dict [str ,Any ]:
        return self .performance_metrics .copy ()

def initialize_ltr_engine (algorithm :RankingAlgorithm =RankingAlgorithm .LISTWISE ,
feature_dim :int =None ,hidden_dims :List [int ]=None ,
learning_rate :float =None ,batch_size :int =None ,
num_epochs :int =None ,device :str ="cpu")->LTRRankingEngine :
    return LTRRankingEngine (
    algorithm =algorithm ,
    feature_dim =feature_dim ,
    hidden_dims =hidden_dims ,
    learning_rate =learning_rate ,
    batch_size =batch_size ,
    num_epochs =num_epochs ,
    device =device 
    )

_ltr_engine =None 

def get_ltr_engine ()->LTRRankingEngine :
    global _ltr_engine 
    if _ltr_engine is None :
        _ltr_engine =initialize_ltr_engine ()
    return _ltr_engine 

class MCPMessageType (Enum ):
    CONTEXT_REQUEST ="context_request"
    CONTEXT_RESPONSE ="context_response"
    AGENT_CALL ="agent_call"
    AGENT_RESPONSE ="agent_response"
    NOTIFICATION ="notification"
    ERROR ="error"

@dataclass 
class MCPMessage :
    message_id :str 
    message_type :MCPMessageType 
    sender_id :str 
    recipient_id :str 
    payload :Dict [str ,Any ]
    timestamp :str 
    context_id :Optional [str ]=None 

    def to_dict (self )->Dict [str ,Any ]:
        return {
        "message_id":self .message_id ,
        "message_type":self .message_type .value ,
        "sender_id":self .sender_id ,
        "recipient_id":self .recipient_id ,
        "payload":self .payload ,
        "timestamp":self .timestamp ,
        "context_id":self .context_id 
        }

class MCPIntegrationManager :
    def __init__ (self ,config :Dict [str ,Any ]=None ):
        self .config =config or {}
        self .active_connections :Dict [str ,Any ]={}
        self .message_handlers :Dict [MCPMessageType ,Callable ]={}
        self .context_store :Dict [str ,Dict [str ,Any ]]={}
        self .message_history :List [MCPMessage ]=[]

        self ._register_default_handlers ()

        logger .info ("MCP integration manager initialized")

    def _register_default_handlers (self ):
        self .message_handlers [MCPMessageType .CONTEXT_REQUEST ]=self ._handle_context_request 
        self .message_handlers [MCPMessageType .AGENT_CALL ]=self ._handle_agent_call 
        self .message_handlers [MCPMessageType .NOTIFICATION ]=self ._handle_notification 

    async def send_message (self ,message :MCPMessage )->Dict [str ,Any ]:
        try :
            self .message_history .append (message )

            handler =self .message_handlers .get (message .message_type )
            if handler :
                response =await handler (message )
                return response 
            else :
                logger .warning (f"No handler for message type: {message .message_type }")
                return {"status":"error","message":"No handler available"}

        except Exception as e :
            logger .error (f"Failed to send MCP message: {e }")
            return {"status":"error","message":str (e )}

    async def _handle_context_request (self ,message :MCPMessage )->Dict [str ,Any ]:
        try :
            context_id =message .payload .get ("context_id")
            if context_id and context_id in self .context_store :
                return {
                "status":"success",
                "context":self .context_store [context_id ]
                }
            else :
                return {
                "status":"error",
                "message":"Context not found"
                }
        except Exception as e :
            return {"status":"error","message":str (e )}

    async def _handle_agent_call (self ,message :MCPMessage )->Dict [str ,Any ]:
        try :
            agent_id =message .payload .get ("agent_id")
            task_data =message .payload .get ("task_data",{})

            if not agent_id :
                raise ValueError ("Agent ID is required for processing")

            result ={
            "agent_id":agent_id ,
            "result":f"Task routed to {agent_id } for processing",
            "task_data":task_data ,
            "processing_status":"completed",
            "timestamp":datetime .now ().isoformat ()
            }

            return {"status":"success","result":result }

        except Exception as e :
            return {"status":"error","message":str (e )}

    async def _handle_notification (self ,message :MCPMessage )->Dict [str ,Any ]:
        try :
            notification_type =message .payload .get ("type")
            notification_data =message .payload .get ("data",{})

            logger .info (f"Received notification: {notification_type }")

            return {
            "status":"success",
            "message":"Notification received"
            }

        except Exception as e :
            return {"status":"error","message":str (e )}

    def store_context (self ,context_id :str ,context_data :Dict [str ,Any ]):
        self .context_store [context_id ]=context_data 
        logger .debug (f"Stored context: {context_id }")

    def get_context (self ,context_id :str )->Optional [Dict [str ,Any ]]:
        return self .context_store .get (context_id )

    def get_message_history (self ,limit :int =100 )->List [MCPMessage ]:
        return self .message_history [-limit :]

class APICallManager :
    def __init__ (self ,config :Dict [str ,Any ]=None ):
        self .config =config or {}
        self .call_history :List [Dict [str ,Any ]]=[]
        self .performance_metrics :Dict [str ,List [float ]]=defaultdict (list )

        logger .info ("API call manager initialized")

    async def make_robust_api_call (self ,api_function :Callable ,max_retries :int =3 ,
    base_delay :float =1.0 ,backoff_factor :float =2.0 ,
    timeout :float =30.0 ,**kwargs )->Dict [str ,Any ]:

        last_exception =None 
        start_time =time .time ()

        for attempt in range (max_retries +1 ):
            try :
                if attempt >0 :
                    delay =base_delay *(backoff_factor **(attempt -1 ))
                    logger .info (f"API call retry {attempt }/{max_retries }, delaying {delay :.2f} seconds...")
                    await asyncio .sleep (delay )
                try :
                    result =await asyncio .wait_for (
                    api_function (**kwargs ),
                    timeout =timeout 
                    )
                    execution_time =time .time ()-start_time 
                    self ._record_api_call (api_function .__name__ ,execution_time ,True ,attempt )

                    return {
                    'status':'success',
                    'data':result ,
                    'attempts':attempt +1 ,
                    'execution_time':execution_time 
                    }

                except asyncio .TimeoutError :
                    raise Exception (f"API call timeout after {timeout } seconds")

            except Exception as e :
                last_exception =e 
                logger .warning (f"API call attempt {attempt +1 } failed: {str (e )}")

                if attempt ==max_retries :
                    execution_time =time .time ()-start_time 
                    self ._record_api_call (api_function .__name__ ,execution_time ,False ,attempt +1 )

                    return {
                    'status':'error',
                    'error':str (last_exception ),
                    'attempts':attempt +1 ,
                    'execution_time':execution_time 
                    }

        return {
        'status':'error',
        'error':'Unknown error occurred',
        'attempts':max_retries +1 ,
        'execution_time':time .time ()-start_time 
        }

    def _record_api_call (self ,function_name :str ,execution_time :float ,
    success :bool ,attempts :int ):
        try :
            call_record ={
            'function_name':function_name ,
            'execution_time':execution_time ,
            'success':success ,
            'attempts':attempts ,
            'timestamp':datetime .now ().isoformat ()
            }

            self .call_history .append (call_record )
            self .performance_metrics [function_name ].append (execution_time )

        except Exception as e :
            logger .warning (f"Failed to record API call metrics: {e }")

    def get_api_performance_summary (self )->Dict [str ,Any ]:
        try :
            summary ={}

            for function_name ,execution_times in self .performance_metrics .items ():
                if execution_times :
                    summary [function_name ]={
                    'total_calls':len (execution_times ),
                    'avg_execution_time':np .mean (execution_times ),
                    'min_execution_time':min (execution_times ),
                    'max_execution_time':max (execution_times ),
                    'success_rate':len ([call for call in self .call_history 
                    if call ['function_name']==function_name and call ['success']])/len (execution_times )
                    }

            return summary 

        except Exception as e :
            logger .error (f"Failed to generate API performance summary: {e }")
            return {}

_mcp_manager =None 
_api_manager =None 

def get_mcp_manager ()->MCPIntegrationManager :
    global _mcp_manager 
    if _mcp_manager is None :
        _mcp_manager =MCPIntegrationManager ()
    return _mcp_manager 

def get_api_manager ()->APICallManager :
    global _api_manager 
    if _api_manager is None :
        _api_manager =APICallManager ()
    return _api_manager 

def create_vrl_system (config :Dict [str ,Any ]=None )->VRL :
    return VRL ()

def create_aep_repository (config :Dict [str ,Any ]=None )->AEPRepository :
    return AEPRepository ()

def create_sbert_engine (config :Dict [str ,Any ]=None )->SBERTSimilarityEngine :
    return SBERTSimilarityEngine ()

def create_marl_engine (config :Dict [str ,Any ]=None )->TrustAwareMARLEngine :
    return TrustAwareMARLEngine (config =config )

def create_ltr_engine (config :Dict [str ,Any ]=None )->LTRRankingEngine :
    return LTRRankingEngine ()

__all__ =[
"TrustRecord",
"TrustDimension",
"VerifiableReputationLedger",
"AEPEntry",
"AEPRepository",
"SBERTSimilarityEngine",
"MARLState",
"AgentQTable",
"TrustAwareMARLEngine",
"LTRFeature",
"LTRNeuralNetwork",
"LTRRankingEngine",
"MCPMessage",
"MCPMessageType",
"MCPIntegrationManager",
"APICallManager",
"create_vrl_system",
"create_aep_repository",
"create_sbert_engine",
"create_marl_engine",
"create_ltr_engine",
"get_ltr_engine",
"get_mcp_manager",
"get_api_manager"
]
