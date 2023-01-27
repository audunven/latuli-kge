from neo4j import GraphDatabase
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models.predict import get_tail_prediction_df
from pykeen.models.predict import predict_triples_df

#the usual password
host = 'bolt://localhost:7687'
user = 'neo4j'
password = ''
driver = GraphDatabase.driver(host, auth=(user, password))


def run_query(query, params={}):
    with driver.session(database='m3v2') as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

data = run_query("""
MATCH (s)-[r]->(t)
RETURN toString(id(s)) as source, toString(id(t)) AS target, type(r) as type
""")

print(data)

tf = TriplesFactory.from_labeled_triples(
  data[["source", "type", "target"]].values,
  create_inverse_triples=False,
  entity_to_id=None,
  relation_to_id=None,
  compact_id=False,
  filter_out_candidate_inverse_relations=True,
  metadata=None,
)

training, testing, validation = tf.split([.8, .1, .1])

#RotatE
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='RotatE',
    stopper='early',
    epochs=20,
    dimensions=512,
    random_seed=420

)

result.save_to_directory('embeddings_rotatE')

data = run_query("""
MATCH (h:Hub)
WHERE h.Date > date("2022-01-01")
RETURN toString(id(h)) AS Id, h.Id as HubId, h.RelativeMaxCapacity as RelativeToMaxCap
""")

print(data)
data.to_csv('embeddings_rotatE/neo4j_data.csv')

#get internal id of HighCapacityPrediction (tail_id=14, tail_label=1093)
highCapacityPrediction_id = run_query("""
MATCH (s:CapacityPrediction)
WHERE s.Id = "HighCapacityPrediction"
RETURN toString(id(s)) as id
""")['id'][0]

#get internal id of LowCapacityPrediction (tail_id=13, tail_label=1092)
lowCapacityPrediction_id = run_query("""
MATCH (s:CapacityPrediction)
WHERE s.Id = "LowCapacityPrediction"
RETURN toString(id(s)) as id
""")['id'][0]

#output csv holding triplet classification of high capacity predictions, the csv will be used for evaluating the kge model
df_high = []
for i in data.index:
    df_high_prediction = predict_triples_df(
        model=result.model,
        triples=(data['Id'][i], 'hasCapacityPrediction', highCapacityPrediction_id),
        triples_factory=result.training,
    )
    df_high.append(df_high_prediction)
df_high = pd.concat(df_high)

print(df_high)

#print df_high to csv
df_high.to_csv('embeddings_rotatE/df_high.csv')

#output csv holding triplet classification of low capacity predictions, the csv will be used for evaluating the kge model
df_low = []
for i in data.index:
    df_low_prediction = predict_triples_df(
        model=result.model,
        triples=(data['Id'][i], 'hasCapacityPrediction', lowCapacityPrediction_id),
        triples_factory=result.training,
    )
    df_low.append(df_low_prediction)
df_low = pd.concat(df_low)

print(df_low)

#print df_high to csv
df_low.to_csv('embeddings_rotatE/df_low.csv')


# #ComplEx
# result = pipeline(
#     training=training,
#     testing=testing,
#     validation=validation,
#     model='ComplEx',
#     stopper='early',
#     epochs=20,
#     dimensions=512,
#     random_seed=420
#
# )
#
# result.save_to_directory('embeddings_ComplEx')
#
# data = run_query("""
# MATCH (h:Hub)
# WHERE h.Date > date("2022-01-01")
# RETURN toString(id(h)) AS Id, h.Id as HubId, h.RelativeMaxCapacity as RelativeToMaxCap
# """)
#
# print(data)
# data.to_csv('embeddings_ComplEx/neo4j_data.csv')
#
# #get internal id of HighCapacityPrediction (tail_id=14, tail_label=1093)
# highCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "HighCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #get internal id of LowCapacityPrediction (tail_id=13, tail_label=1092)
# lowCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "LowCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #output csv holding triplet classification of high capacity predictions, the csv will be used for evaluating the kge model
# df_high = []
# for i in data.index:
#     df_high_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', highCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_high.append(df_high_prediction)
# df_high = pd.concat(df_high)
#
# print(df_high)
#
# #print df_high to csv
# df_high.to_csv('embeddings_ComplEx/df_high.csv')
#
# #output csv holding triplet classification of low capacity predictions, the csv will be used for evaluating the kge model
# df_low = []
# for i in data.index:
#     df_low_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', lowCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_low.append(df_low_prediction)
# df_low = pd.concat(df_low)
#
# print(df_low)
#
# #print df_high to csv
# df_low.to_csv('embeddings_ComplEx/df_low.csv')

# #TransE
# result = pipeline(
#     training=training,
#     testing=testing,
#     validation=validation,
#     model='TransE',
#     stopper='early',
#     epochs=20,
#     dimensions=512,
#     random_seed=420
#
# )
#
# result.save_to_directory('embeddings_TransE')
#
# data = run_query("""
# MATCH (h:Hub)
# WHERE h.Date > date("2022-01-01")
# RETURN toString(id(h)) AS Id, h.Id as HubId, h.RelativeMaxCapacity as RelativeToMaxCap
# """)
#
# print(data)
# data.to_csv('embeddings_TransE/neo4j_data.csv')
#
# #get internal id of HighCapacityPrediction (tail_id=14, tail_label=1093)
# highCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "HighCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #get internal id of LowCapacityPrediction (tail_id=13, tail_label=1092)
# lowCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "LowCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #output csv holding triplet classification of high capacity predictions, the csv will be used for evaluating the kge model
# df_high = []
# for i in data.index:
#     df_high_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', highCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_high.append(df_high_prediction)
# df_high = pd.concat(df_high)
#
# print(df_high)
#
# #print df_high to csv
# df_high.to_csv('embeddings_TransE/df_high.csv')
#
# #output csv holding triplet classification of low capacity predictions, the csv will be used for evaluating the kge model
# df_low = []
# for i in data.index:
#     df_low_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', lowCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_low.append(df_low_prediction)
# df_low = pd.concat(df_low)
#
# print(df_low)
#
# #print df_high to csv
# df_low.to_csv('embeddings_TransE/df_low.csv')


# #TuckER
# result = pipeline(
#     training=training,
#     testing=testing,
#     validation=validation,
#     model='TuckER',
#     stopper='early',
#     epochs=20,
#     dimensions=512,
#     random_seed=420
#
# )
#
# result.save_to_directory('embeddings_TuckER')
#
# data = run_query("""
# MATCH (h:Hub)
# WHERE h.Date > date("2022-01-01")
# RETURN toString(id(h)) AS Id, h.Id as HubId, h.RelativeMaxCapacity as RelativeToMaxCap
# """)
#
# print(data)
# data.to_csv('embeddings_TuckER/neo4j_data.csv')
#
# #get internal id of HighCapacityPrediction (tail_id=14, tail_label=1093)
# highCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "HighCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #get internal id of LowCapacityPrediction (tail_id=13, tail_label=1092)
# lowCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "LowCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #output csv holding triplet classification of high capacity predictions, the csv will be used for evaluating the kge model
# df_high = []
# for i in data.index:
#     df_high_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', highCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_high.append(df_high_prediction)
# df_high = pd.concat(df_high)
#
# print(df_high)
#
# #print df_high to csv
# df_high.to_csv('embeddings_TuckER/df_high.csv')
#
# #output csv holding triplet classification of low capacity predictions, the csv will be used for evaluating the kge model
# df_low = []
# for i in data.index:
#     df_low_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', lowCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_low.append(df_low_prediction)
# df_low = pd.concat(df_low)
#
# print(df_low)
#
# #print df_high to csv
# df_low.to_csv('embeddings_TuckER/df_low.csv')

# #ConvE
# result = pipeline(
#     training=training,
#     testing=testing,
#     validation=validation,
#     model='ConvE',
#     stopper='early',
#     epochs=20,
#     dimensions=512,
#     random_seed=420
#
# )
#
# result.save_to_directory('embeddings_ConvE')
#
# data = run_query("""
# MATCH (h:Hub)
# WHERE h.Date > date("2022-01-01")
# RETURN toString(id(h)) AS Id, h.Id as HubId, h.RelativeMaxCapacity as RelativeToMaxCap
# """)
#
# print(data)
# data.to_csv('embeddings_ConvE/neo4j_data.csv')
#
# #get internal id of HighCapacityPrediction (tail_id=14, tail_label=1093)
# highCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "HighCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #get internal id of LowCapacityPrediction (tail_id=13, tail_label=1092)
# lowCapacityPrediction_id = run_query("""
# MATCH (s:CapacityPrediction)
# WHERE s.Id = "LowCapacityPrediction"
# RETURN toString(id(s)) as id
# """)['id'][0]
#
# #output csv holding triplet classification of high capacity predictions
# df_high = []
# for i in data.index:
#     df_high_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', highCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_high.append(df_high_prediction)
# df_high = pd.concat(df_high)
#
# print(df_high)
#
# #print df_high to csv
# df_high.to_csv('embeddings_ConvE/df_high.csv')
#
# #output csv holding triplet classification of low capacity predictions
# df_low = []
# for i in data.index:
#     df_low_prediction = predict_triples_df(
#         model=result.model,
#         triples=(data['Id'][i], 'hasCapacityPrediction', lowCapacityPrediction_id),
#         triples_factory=result.training,
#     )
#     df_low.append(df_low_prediction)
# df_low = pd.concat(df_low)
#
# print(df_low)
#
# #print df_high to csv
# df_low.to_csv('embeddings_ConvE/df_low.csv')
