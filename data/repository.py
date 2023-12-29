"""
A wrapper for sqlite3 database
"""
import csv
import io
import os
import sqlite3
import time
from typing import Iterable, Dict, Union

import numpy as np


def np_to_db(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def db_to_np(text):
    if text is None:
        return None
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def get_connection(deploy=False, result_path="result"):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if deploy:
        return sqlite3.Connection(os.path.join(result_path, "data_deploy.db"))
    else:
        return sqlite3.Connection(os.path.join(result_path, "data.db"))


def execute_sql(conn: sqlite3.Connection, sql: str, parameters: Iterable = None):
    # print(sql, parameters, end='   ')
    start = time.time()
    try:
        if parameters is not None:
            result = conn.execute(sql, parameters)
        else:
            result = conn.execute(sql)
    except sqlite3.OperationalError as e:
        print(sql, parameters)
        raise e
    # print(time.time() - start, 's')
    return result


def execute_sql_return_first(conn: sqlite3.Connection, sql: str, parameters: Iterable = None):
    cursor = execute_sql(conn, sql, parameters)
    for x in cursor:
        return x
    return None


def execute_many(conn: sqlite3.Connection, sql: str, seq_of_parameters: list = None):
    # print("sql: %s [%d]" % (sql, len(seq_of_parameters)))
    conn.executemany(sql, seq_of_parameters)


def create_table(conn: sqlite3.Connection, table_name: str, columns: Dict,
                 primary_keys: list = None):
    columns = ["%s %s" % (k, v) for k, v in columns.items()]
    if primary_keys is not None:
        columns.append("PRIMARY KEY (%s)" % ", ".join(primary_keys))
    sql = "CREATE TABLE IF NOT EXISTS `%s` (%s)" % (
        table_name,
        ", ".join(columns)
    )
    execute_sql(conn, sql)


def create_index(conn: sqlite3.Connection, index_name: str, table_name: str, columns: list):
    sql = "CREATE INDEX %s on `%s` (%s)" % (
        index_name,
        table_name,
        ", ".join(columns)
    )
    try:
        execute_sql(conn, sql)
    except Exception:
        pass


def get_columns(conn: sqlite3.Connection, table_name: str):
    cursor = select(conn, table_name)
    return get_columns_by_cursor(cursor)


def get_columns_by_cursor(cursor):
    descriptions = list(cursor.description)
    return [description[0] for description in descriptions]


def ensure_column(conn: sqlite3.Connection, table_name: str,
                  name_type_default: Iterable):
    columns = get_columns(conn, table_name)
    for name, db_type, default in name_type_default:
        if name not in columns:
            if default is not None:
                statement = ("ALTER TABLE %s ADD COLUMN %s %s DEFAULT `%s`" % (
                    table_name, name, db_type, default
                ))
            else:
                statement = ("ALTER TABLE %s ADD COLUMN %s %s" % (
                    table_name, name, db_type
                ))
            execute_sql(conn, statement)


def update(conn: sqlite3.Connection, table_name: str, puts: list,
           wheres: list):
    sql = None
    assert len(puts) == len(wheres)
    assert len(puts) != 0
    parameters = []
    for put, where in zip(puts, wheres):
        replace_items = sorted(put.items(), key=lambda x: x[0])
        where_items = sorted(where.items(), key=lambda x: x[0])

        if sql is None:
            sql = 'UPDATE `%s`' % table_name + \
                  ' SET ' + ', '.join(['%s = ?' % k for k, _ in replace_items]) + \
                  ' WHERE ' + " AND ".join(['%s = ?' % k for k, _ in where_items])

        parameter = [v for _, v in replace_items] + [v for _, v in where_items]
        parameters.append(parameter)

    execute_many(conn, sql, parameters)


def delete(conn: sqlite3.Connection, table_name: str, where: dict):
    where_items = where.items()

    sql = 'DELETE FROM `%s`' % table_name + \
          ' WHERE ' + " AND ".join(['%s = ?' % k for k, _ in where_items])

    parameters = [v for _, v in where_items]

    return execute_sql(conn, sql, parameters)


def select(conn: sqlite3.Connection, table_name: Union[str, list, tuple],
           project: list = None, where: dict = None,
           order_by: str = None, limit: int = None,
           offset: int = None,
           return_first: bool = False, prefix: str = "",
           where_format="%s = ?", group_by: list = None):
    if where is None or len(where) == 0:
        where = {"1": 1}
    where_items = where.items()
    project = ", ".join(project) if project is not None else "*"
    if type(table_name) is list or type(table_name) is tuple:
        table_name = ", ".join(table_name)
    sql = prefix + "SELECT %s FROM %s" % (project, table_name) + \
          " WHERE " + " AND ".join([where_format % k for k, _ in where_items])
    if order_by is not None:
        sql += " ORDER BY " + order_by
    if limit is not None:
        sql += " LIMIT " + str(limit)
    if offset is not None:
        sql += f" OFFSET {offset}"
    if group_by is not None:
        sql += " GROUP BY %s" % (", ".join(group_by))

    # print(sql)
    parameters = [v for _, v in where_items]
    if return_first:
        return execute_sql_return_first(conn, sql, parameters)
    else:
        return execute_sql(conn, sql, parameters)


def select_first(conn: sqlite3.Connection, table_name: Union[str, list, tuple],
                 project: list = None, where: dict = None,
                 order_by: str = None, limit: int = None):
    cursor = select(conn, table_name, project, where, order_by, limit)
    for x in cursor:
        return x
    return None


def count(conn: sqlite3.Connection, table_name: str, where: dict = None):
    return select_first(conn, table_name, project=["COUNT(*)"], where=where)[0]


def insert_or_replace(conn: sqlite3.Connection, table_name: str, contents: list, or_ignore=False):
    if len(contents) == 0:
        return
    columns = contents[0].keys()
    sql = "INSERT OR " + ("IGNORE" if or_ignore else "REPLACE") + " INTO `%s` (%s) VALUES (%s)" % (
        table_name, ", ".join(columns),
        ", ".join(["?"] * (len(columns)))
    )

    seq_of_params = [
        [model[column] for column in columns]
        for model in contents
    ]
    execute_many(conn, sql, seq_of_params)


def export_db_to_csv(table_name, out_csv_name):
    with get_connection() as conn:
        with open(os.path.join("result", out_csv_name), "w") as f:
            fcsv = csv.writer(f)
            cursor = select(conn, table_name)
            fcsv.writerow([i[0] for i in cursor.description])
            fcsv.writerows(cursor.fetchall())
