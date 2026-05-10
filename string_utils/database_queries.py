# ====================
# langgraph_core.py

class Queries:
    ## === Tool: put_new_memory

    CHECK_MEMORY = """
    SELECT memory_id
    FROM "User_memory"
    WHERE user_id = $1 AND tenant_id = $2;
    """

    PUT_NEW_MEMORY = """
    INSERT INTO "User_memory" (memory_id, tenant_id, user_id, content, embedding)
    VALUES ($1, $2, $3, $4, $5)
    """

    ## === Tool: fetch_new_knowledge

    FETCH_NEW_KNOWLEDGE_CHUNK = """
    SELECT chunk_id, knowledge_id, content
    FROM "Knowledge_vectors"
    WHERE tenant_id = $1
    ORDER BY embedding <=> $2
    LIMIT 5;
    """

    FETCH_NEW_KNOWLEDGE = """
    SELECT knowledge_id, metadata
    FROM "Knowledges"
    WHERE tenant_id = $1 AND knowledge_id = ANY($2);
    """

    ## === Tool: fetch_new_memory

    FETCH_NEW_MEMORY = """
    SELECT memory_id, content
    FROM "User_memory"
    WHERE user_id = $1 AND tenant_id = $2
    ORDER BY embedding <=> $3
    LIMIT 5;
    """

    ## === Tool: fetch_new_knowledge_session

    FETCH_NEW_KNOWLEDGE_SESSION_CHUNK = """
    SELECT chunk_id, s_knowledge_id, content
    FROM "Session_vectors"
    WHERE tenant_id = $1 AND user_id = $2 AND thread_id = $3
    ORDER BY embedding <=> $4
    LIMIT 5;
    """

    FETCH_NEW_KNOWLEDGE_SESSION = """
    SELECT s_knowledge_id, metadata
    FROM "Session_knowledges"
    WHERE tenant_id = $1 AND s_knowledge_id = ANY($2) AND user_id = $3 AND thread_id = $4;
    """

    ## === Node: check_knowledge_session_ttl

    QUERY_CHECK_KNOWLEDGE_SESSION_TTL = """
    SELECT s_knowledge_id
    FROM "Session_knowledges"
    WHERE s_knowledge_id = ANY($1) AND tenant_id = $2 AND user_id = $3 AND thread_id = $4;
    """

    ## === Node: fetch_knowledge_session

    QUERY_FETCH_KNOWLEDGE_SESSION = """
    SELECT s_knowledge_id, chunk_id, content
    FROM "Session_vectors"
    WHERE chunk_id = ANY($1) AND tenant_id = $2 AND user_id = $3 AND thread_id = $4;
    """

    ## === Node: check_knowledge_exist

    QUERY_CHECK_KNOWLEDGE_EXIST = """
    SELECT knowledge_id
    FROM "Knowledges"
    WHERE knowledge_id = ANY($1) AND tenant_id = $2;
    """

    ## === Node: fetch_knowledge

    QUERY_FETCH_KNOWLEDGE = """
    SELECT knowledge_id, chunk_id, content
    FROM "Knowledge_vectors"
    WHERE chunk_id = ANY($1) AND tenant_id = $2;
    """

    ## === Node: check_memory_exist_and_fetch

    QUERY_MEMORY_EXIST = """
    SELECT memory_id, content
    FROM "User_memory"
    WHERE memory_id = ANY($1) AND tenant_id = $2 AND user_id = $3;
    """

    ## === Node: rag

    FETCH_CHUNKS = """
    SELECT chunk_id, knowledge_id, content
    FROM "Knowledge_vectors"
    WHERE tenant_id = $1
    ORDER BY embedding <=> $2
    LIMIT 5;
    """

    FETCH_KNOWLEDGE = """
    SELECT metadata
    FROM "Knowledges"
    WHERE tenant_id = $1 AND knowledge_id = ANY($2)
    ORDER BY array_position($2, knowledge_id);
    """

# ====================
# document_utils.py

class DocumentQueries:
    ## === Function: save_chunks_session_to_db
    
    INPUT_SESSION_KNOWLEDGES = """
    INSERT INTO "Session_nowledges" (s_knowledge_id, tenant_id, user_id, metadata)
    VALUES ($1, $2, $3, $4)
    """

    INPUT_SESSION_KNOWLEDGE_CHUNKS = """
    INSERT INTO "Session_vectors" (chunk_id, s_knowledge_id, tenant_id, user_id, content, embedding)
    VALUES ($1, $2, $3, $4, $5, $6)
    """

    ## === Function: save_chunks_to_db

    INPUT_KNOWLEDGES = """
    INSERT INTO "Knowledges" (knowledge_id, tenant_id, metadata)
    VALUES ($1, $2, $3)
    """

    INPUT_KNOWLEDGE_CHUNKS = """
    INSERT INTO "Knowledge_vectors" (chunk_id, knowledge_id, tenant_id, content, embedding)
    VALUES ($1, $2, $3, $4, $5)
    """

# ====================
# chat_completion.py
    
class TitleQueries:
    ## === Function: new_chat
    
    FETCH_SESSION_TITLE = """
    SELECT thread_id, title
    FROM "Sessions"
    WHERE thread_id = $1;
    """
    
    INSERT_SESSION_TITLE = """
    INSERT INTO "Sessions" (thread_id, tenant_id, user_id, title)
    VALUES ($1, $2, $3)
    """
    
# ====================
# add_member.py

class AddQueries:
    ## === Function: new_company
    
    INSERT_NEW_COMPANY = """
    INSERT INTO "Tenants" (tenant_id, name)
    VALUES ($1, $2)
    """
    
    ## === Function: new_user
    
    INSERT_NEW_USER = """
    INSERT INTO "Users" (user_id, tenant_id, role, name)
    VALUES ($1, $2, $3, $4)
    """