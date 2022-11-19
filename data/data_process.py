import numpy as np
from scipy.stats import ortho_group

from data.model import *
import random
import time


def ensure_embedding_column(conn, table_name, embedding_name, config: NetworkConfig):
    """
    Ensure the embedding column in the database. Each value in the embedding vector takes a column.
    :param conn: database connection
    :param table_name: table name
    :param embedding_name: base embedding name. The i-th value is saved in {embedding_name}_i
    :param config: training config
    :return: nothing
    """
    columns = [(column, 'REAL', None) for column in
               (config.get_embedding_names(embedding_name) +
                [config.get_embedding_names(embedding_name, is_alpha=True)])
               ]
    columns += [(config.get_embedding_names(embedding_name, is_sigma=True), 'BLOB', None)]
    repository.ensure_column(conn, table_name=table_name,
                             name_type_default=columns)


def seed(source):
    """
    seed everything to keep consistency
    :param source: seed source
    """
    s = abs(hash(source)) % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def load_embedding(table_name, primary_keys, embedding_name, config: NetworkConfig,
                   initializer=None):
    """
    load EmbeddingData from database
    :param table_name: table name
    :param primary_keys: a list storing the primary key columns
    :param embedding_name: base embedding column name
    :param config: training config
    :param initializer: a function return a value for an embedding value when it is not initialized
    :return: an EmbeddingData
    """
    # db key -> emb id
    key_to_embed_id = {}
    if initializer is None:
        def initializer(primary_values):
            arr = (np.random.random((embedding_size,)) + 0.1) * np.sign(np.random.random((embedding_size,)) - 0.5)
            # arr[-1] = 0.5 / embedding_size
            return arr
    embeddings = []
    sigmas = []
    alphas = []
    embedding_size = config.embedding_size

    project = list(primary_keys)
    project += config.get_embedding_names(embedding_name)
    project.append(config.get_embedding_names(embedding_name, is_sigma=True))
    project.append(config.get_embedding_names(embedding_name, is_alpha=True))
    with repository.get_connection() as conn:
        ensure_embedding_column(conn, table_name, embedding_name, config)
        cursor = repository.select(conn, table_name=table_name, project=project)
    for i, tpl in enumerate(cursor):
        primary_values = tpl[:len(primary_keys)]
        seed(tuple(primary_values))
        embedding = list(tpl[len(primary_keys):len(primary_keys) + embedding_size])
        sigma = repository.db_to_np(tpl[-2])
        alpha = tpl[-1]

        for j in range(embedding_size):
            if embedding[j] is None:
                embedding = initializer(primary_values)
                break
        embedding = np.asarray(embedding)
        embeddings.append(embedding)
        sigmas.append(sigma if (sigma is not None and sigma.shape == (embedding_size, embedding_size))
                      else np.zeros((embedding_size, embedding_size), np.float64))
        alphas.append(alpha if alpha is not None else 0.0)
        key_to_embed_id["-".join(map(str, primary_values))] = i
    return EmbeddingData(
        key_to_embed_id=pd.Series(key_to_embed_id, name=table_name),
        embeddings=[np.asarray(embeddings)],
        sigma=np.asarray(sigmas, np.float64),
        alpha=np.asarray(alphas, np.float64)
    )


def load_user_embedding_online(table_name, primary_keys, embedding_name, config: NetworkConfig, user_key, connection,
                               initializer=None):
    embedding_size = config.embedding_size
    return EmbeddingData(
        key_to_embed_id=pd.Series({user_key: 0}, name=table_name),
        embeddings=[np.zeros((1, embedding_size))],
        sigma=np.zeros((1, embedding_size, embedding_size), dtype=np.float64),
        alpha=np.zeros((1, ), dtype=np.float64)
    )


def load_beatmap_embedding_online(table_name, primary_keys, embedding_name, config: NetworkConfig, user_key, connection,
                                  ):
    key_to_embed_id = {}
    embeddings = []
    sigmas = []
    alphas = []
    embedding_size = config.embedding_size
    user_id, game_mode, variant = user_key.split("-")

    project = list(primary_keys)
    project += config.get_embedding_names(embedding_name)
    project.append(config.get_embedding_names(embedding_name, is_sigma=True))
    project.append(config.get_embedding_names(embedding_name, is_alpha=True))

    sql = (
        f"SELECT s.{Score.BEATMAP_ID} "
        f"FROM {Score.TABLE_NAME} as s "
        f"WHERE s.{Score.GAME_MODE} == '{game_mode}' "
        f"AND s.{Score.USER_ID} == {user_id} "
        f"AND s.cs == '{variant[0]}' "
        f"AND s.{Score.PP} >= 1 "
        f"AND NOT s.{Score.IS_EZ} ")

    for k, x in enumerate(repository.execute_sql(connection, sql)):
        beatmap_id = int(''.join(map(str, x)))
        constrain = {'id': beatmap_id}
        cursor = repository.select(connection, table_name=table_name, project=project, where=constrain)

        for i, tpl in enumerate(cursor):
            primary_values = tpl[:len(primary_keys)]
            seed(tuple(primary_values))
            embedding = list(tpl[len(primary_keys):len(primary_keys) + embedding_size])
            sigma = repository.db_to_np(tpl[-2])
            alpha = tpl[-1]

            if embedding[0] is None:
                break
            if str(beatmap_id) in key_to_embed_id.keys():
                break

            embedding = np.asarray(embedding)
            embeddings.append(embedding)
            sigmas.append(sigma if (sigma is not None and sigma.shape == (embedding_size, embedding_size))
                          else np.zeros((embedding_size, embedding_size), np.float64))
            alphas.append(alpha if alpha is not None else 0.0)
            key_to_embed_id["-".join(map(str, primary_values))] = len(embeddings) - 1
    return EmbeddingData(
        key_to_embed_id=pd.Series(key_to_embed_id, name=table_name),
        embeddings=[np.asarray(embeddings)],
        sigma=np.asarray(sigmas, np.float64),
        alpha=np.asarray(alphas, np.float64)
    )


def save_embedding(conn, embedding: EmbeddingData, config: NetworkConfig,
                   table_name: str, embedding_name: str):
    """
    Save embedding into database. Please use it with load_embedding
    :param conn: database connection
    :param embedding: EmbeddingData to be saved
    :param config: training config
    :param table_name: table name
    :param embedding_name: base embedding column name
    :return: nothing
    """
    puts = []
    wheres = []
    ensure_embedding_column(conn, table_name, embedding_name, config)

    def embedding_array_to_db_dict(embedding_array, key, alpha=None, sigma: np.ndarray = None):
        puts = dict(zip(config.get_embedding_names(key), embedding_array))
        if alpha is not None:
            puts[config.get_embedding_names(key, is_alpha=True)] = float(alpha)
        if sigma is not None:
            puts[config.get_embedding_names(key, is_sigma=True)] = repository.np_to_db(sigma.astype(np.float32))
        return puts

    for key, emb_id in embedding.key_to_embed_id.items():
        weights = embedding.embeddings[0][emb_id]
        alpha = sigma = None
        if embedding.alpha is not None:
            alpha = embedding.alpha[emb_id]
        if embedding.sigma is not None:
            sigma = embedding.sigma[emb_id]
        puts.append(embedding_array_to_db_dict(weights, embedding_name, alpha, sigma))
        if table_name == BeatmapEmbedding.TABLE_NAME:
            wheres.append({
                BeatmapEmbedding.BEATMAP_ID: key
                # BeatmapEmbedding.SPEED: beatmap_speed
            })
        elif table_name == UserEmbedding.TABLE_NAME:
            wheres.append(UserEmbedding.construct_where_with_key(key))
        elif table_name == ModEmbedding.TABLE_NAME:
            wheres.append({
                ModEmbedding.MOD: key
            })
        else:
            raise ValueError(table_name)

    repository.update(conn, table_name=table_name, puts=puts, wheres=wheres)


@measure_time
def load_weight(config: NetworkConfig):
    """
    load weights (i.e., embedding data) from database
    :param config: training config
    :return: ScoreModelWeight
    """
    with repository.get_connection() as connection:
        BeatmapEmbedding.create(connection)
        UserEmbedding.create(connection)
        ModEmbedding.create(connection)
        Meta.create(connection)
        # initialize embedding items
        repository.select(connection, User.TABLE_NAME, User.PRIMARY_KEYS,
                          prefix="INSERT OR IGNORE INTO %s (%s) " % (UserEmbedding.TABLE_NAME,
                                                                     ", ".join(
                                                                         UserEmbedding.PRIMARY_KEYS)),
                          where={User.GAME_MODE: config.game_mode})
        repository.select(connection, Beatmap.TABLE_NAME, Beatmap.PRIMARY_KEYS,
                          prefix="INSERT OR IGNORE INTO %s (%s) " % (BeatmapEmbedding.TABLE_NAME,
                                                                     ", ".join(
                                                                         BeatmapEmbedding.PRIMARY_KEYS)),
                          where={Beatmap.GAME_MODE: config.game_mode})
        repository.insert_or_replace(connection, ModEmbedding.TABLE_NAME, [
            {ModEmbedding.MOD: "NM", ModEmbedding.SPEED: '0', ModEmbedding.IS_ACC: False},
            {ModEmbedding.MOD: "HT", ModEmbedding.SPEED: '-1', ModEmbedding.IS_ACC: False},
            {ModEmbedding.MOD: "DT", ModEmbedding.SPEED: '1', ModEmbedding.IS_ACC: False},
            {ModEmbedding.MOD: "NM-ACC", ModEmbedding.SPEED: '0', ModEmbedding.IS_ACC: True},
            {ModEmbedding.MOD: "HT-ACC", ModEmbedding.SPEED: '-1', ModEmbedding.IS_ACC: True},
            {ModEmbedding.MOD: "DT-ACC", ModEmbedding.SPEED: '1', ModEmbedding.IS_ACC: True},
        ], or_ignore=True)

    # load embedding weights
    weights = ScoreModelWeight()
    # if os.path.exists(SCORE_WEIGHT_PATH):
    #     with open(SCORE_WEIGHT_PATH, "rb") as f:
    #         feature_weights = pickle.load(f)
    #         print("[load_weight] old weight keys: " + str(feature_weights.keys()))
    #         weights.feature_weights = feature_weights

    weights.user_embedding = load_embedding(UserEmbedding.TABLE_NAME, UserEmbedding.PRIMARY_KEYS,
                                            UserEmbedding.EMBEDDING, config)
    weights.beatmap_embedding = load_embedding(BeatmapEmbedding.TABLE_NAME,
                                               BeatmapEmbedding.PRIMARY_KEYS,
                                               BeatmapEmbedding.ITEM_EMBEDDING,
                                               config)

    m = ortho_group.rvs(dim=config.embedding_size)

    def mod_embedding_initializer2(primary_values):
        """
        An intializer for mod. This is to make the mod embeddings orthogonal
        :param primary_values: here is the mod name
        :return: initialization value
        """
        w = 0.1
        w2 = 0.5
        nm = m[0]
        ht = m[1] * (1 - w) + nm * w
        dt = m[2] * (1 - w) + nm * w
        nm_acc = m[3] * (1 - w2) + nm * w2
        ht_acc = m[4] * (1 - w2 - w) + ht * w2 + nm_acc * w
        dt_acc = m[5] * (1 - w2 - w) + dt * w2 + nm_acc * w
        if primary_values[0] == 'NM':
            return nm
        elif primary_values[0] == 'HT':
            return ht
        elif primary_values[0] == 'DT':
            return dt
        elif primary_values[0] == 'NM-ACC':
            return nm_acc
        elif primary_values[0] == 'HT-ACC':
            return ht_acc
        elif primary_values[0] == 'DT-ACC':
            return dt_acc
        raise

    weights.mod_embedding = load_embedding(ModEmbedding.TABLE_NAME,
                                           ModEmbedding.PRIMARY_KEYS,
                                           ModEmbedding.EMBEDDING, config,
                                           mod_embedding_initializer2)
    # print("mod corr: ")
    # def cos_sim(a, b):
    #     return np.dot(a, b) / (np.linalg.norm(a) + 1e-6) / (np.linalg.norm(b) + 1e-6)
    # for idx_i, i in enumerate(weights.mod_embedding.key_to_embed_id.keys()):
    #     for idx_j, j in enumerate(weights.mod_embedding.key_to_embed_id.keys()):
    #         if idx_i <= idx_j:
    #             continue
    #         ii = weights.mod_embedding.key_to_embed_id[i]
    #         jj = weights.mod_embedding.key_to_embed_id[j]
    #         sim = cos_sim(weights.mod_embedding.embeddings[0][ii], weights.mod_embedding.embeddings[0][jj])
    #         print(f"{i} <-> {j}: {sim}")

    # mod_embedding = np.zeros((3, config.embedding_size, config.embedding_size), dtype=np.float64)
    # for i in range(3):
    #     for j in range(config.embedding_size // 3):
    #         mod_embedding[i, i + j, i + j]
    # mod_embedding[0, ]
    # weights.mod_embedding = EmbeddingData(
    #     key_to_embed_id=pd.Series({'HT': 0, 'NM': 1, 'DT': 2}),
    #     embeddings=[np.asarray([
    #         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    #     ], dtype=np.float64)],
    #     sigma=np.zeros((3, config.embedding_size, config.embedding_size), np.float64),
    #     alpha=np.zeros((3,), np.float64),
    # )
    return weights


def load_weight_online(config: NetworkConfig, user_key, connection):
    weights = ScoreModelWeight()
    weights.user_embedding = load_user_embedding_online(UserEmbedding.TABLE_NAME,
                                                        UserEmbedding.PRIMARY_KEYS,
                                                        UserEmbedding.EMBEDDING,
                                                        config, user_key, connection)
    weights.beatmap_embedding = load_beatmap_embedding_online(BeatmapEmbedding.TABLE_NAME,
                                                              BeatmapEmbedding.PRIMARY_KEYS,
                                                              BeatmapEmbedding.ITEM_EMBEDDING,
                                                              config, user_key, connection)
    weights.mod_embedding = load_embedding(ModEmbedding.TABLE_NAME,
                                           ModEmbedding.PRIMARY_KEYS,
                                           ModEmbedding.EMBEDDING, config)
    return weights
