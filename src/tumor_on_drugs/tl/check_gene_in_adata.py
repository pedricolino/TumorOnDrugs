"""Gene lookup helpers."""


def check_gene_in_adata(
    gene_name,
    adata,
    return_synonyms=False,
    return_boolean=False,
):
    if return_synonyms and return_boolean:
        raise ValueError(
            "Both return_synonyms and return_boolean cannot be True at the same time."
        )
    if gene_name in adata.var_names:
        print(f"{gene_name} is present in the dataset.")
        if return_boolean:
            return True
        if return_synonyms:
            return [gene_name]
        return None
    from difflib import get_close_matches

    from mygene import MyGeneInfo

    result = MyGeneInfo().query(gene_name, fields="symbol", species="human")
    synonyms = [hit["symbol"] for hit in result.get("hits", [])]
    found = [synonym for synonym in synonyms if synonym in adata.var_names]
    if found:
        print(
            f"{gene_name} is NOT present in the dataset, but the following "
            f"synonyms were found in the dataset: {', '.join(found)}"
        )
        return True if return_boolean else found if return_synonyms else None
    print(f"{gene_name} is NOT present in the dataset and no synonyms were found.")
    close_matches = get_close_matches(gene_name, adata.var_names, n=5)
    if close_matches:
        print(f"Did you mean: {', '.join(close_matches)}?")
    if return_boolean:
        return False
    if return_synonyms:
        return []
    return None
