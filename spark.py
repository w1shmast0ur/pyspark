import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StringType, BooleanType
from pyspark.sql import functions as func

# create spark Session
spark = SparkSession.builder.appName("StructuredStream").getOrCreate()
sc = spark.sparkContext

# define Schema of json
mySchema = StructType() \
    .add('timestamp', 'integer') \
    .add('event_type', 'string') \
    .add('user_id', 'string') \
    .add('content_id', 'string')

# read json
DF = spark.read.schema(mySchema).json('events.json')
DF.printSchema()

# save and load parquete
DF.write.partitionBy('user_id').mode('overwrite').parquet('events.parquet')
DF = spark.read.parquet('events.parquet')

# define partition for users
from pyspark.sql import Window

wUsers = Window.orderBy('content_id', 'timestamp').partitionBy('user_id')

# define sessions for each user
DF = DF.withColumn('session', (DF.event_type == 'stream_start').cast('int'))
DF = DF.withColumn('session', func.sum('session').over(wUsers))
DF.show()


# function to provide session status
def status(event_type):
    statuses = {
        'stream_start': 'open',
        'ad_start': 'open/ad',
        'ad_end': 'open',
        'track_start': 'open/playing',
        'track_hearbeat': 'open/playing',
        'pause': 'open/paused',
        'play': 'open/playing',
        'track_end': 'open/end',
        'stream_end': 'closed'
    }
    return statuses[event_type]


# make it udf and use it in defining of session status
udfStatus = func.udf(status, StringType())
DF = DF.withColumn('status', udfStatus('event_type'))
DF.show()

# define partitions for sesssion and users
wSessions = Window.orderBy('timestamp').partitionBy('user_id', 'session')

# count the events
DF = DF.withColumn('event_count', func.row_number().over(wSessions))
DF.orderBy('session', 'timestamp').show()

# count ad
DF = DF.withColumn('ad_count', (DF.event_type == 'ad_start').cast('int'))
DF = DF.withColumn('ad_count', func.sum('ad_count').over(wSessions))
DF.show()

# count time and total time
DF = DF.withColumn('time', DF.timestamp - func.lag('timestamp', 1).over(wSessions)).na.fill(0)
DF = DF.withColumn('total_time', func.sum('time').over(wSessions))
DF.show()

# count ad time
DF = DF.withColumn('ad_time', func.when(DF['event_type'] == 'ad_end', DF['time'])).na.fill(0)
DF = DF.withColumn('ad_time', func.sum('ad_time').over(wSessions))
DF.orderBy('session', 'timestamp').show()


# check is main content is playing
def checkPlay(event_type):
    statuses = {
        'stream_start': False,
        'ad_start': False,
        'ad_end': False,
        'track_start': False,
        'track_hearbeat': True,
        'pause': True,
        'play': False,
        'track_end': True,
        'stream_end': False
    }
    return statuses[event_type]


udfCheckPlay = func.udf(checkPlay, BooleanType())

# count playing time
DF = DF.withColumn('play_time', func.when(udfCheckPlay(DF['event_type']), DF['time'])).na.fill(0)
DF = DF.withColumn('play_time', func.sum('play_time').over(wSessions))
DF.orderBy('user_id', 'session', 'timestamp').show()

# check which session are closed
DF = DF.withColumn('output',
                   func.when(DF['session'] != func.max('session').over(Window.partitionBy('user_id')), True)
                   .otherwise(False))
DF = DF.withColumn('output', func.when(DF['event_type'] == 'stream_end', True).otherwise(DF.output))
DF = DF.withColumn('output', func.max('output').over(wSessions))
# why this doesn't work?
# DF = DF.withColumn('output',func.max('session').over(w))
DF.orderBy('user_id', 'timestamp').show()

# aggregate result
res = DF.filter(DF.output) \
    .groupby('user_id', 'content_id', 'session') \
    .agg(func.min('timestamp').alias('session_start'),
         func.max('timestamp').alias('session_end'),
         func.max('total_time').alias('total_time'),
         func.max('play_time').alias('track_playtime'),
         func.max('event_count').alias('event_count'),
         func.max('ad_count').alias('ad_count')) \
    .select('user_id',
            'content_id',
            'session_start',
            'session_end',
            'total_time',
            'track_playtime',
            'event_count',
            'ad_count') \
    .orderBy('user_id', 'content_id')
res.show()

res.coalesce(1).write.mode('overwrite').json('result')
