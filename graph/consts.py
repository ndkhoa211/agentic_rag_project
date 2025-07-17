# define constants, aka the node names that we're
# going to refer to in the graph
# the reason we do this to avoid code duplication
# so everytime we reference the nodes we'll
# actually reference the const, and if we want to
# change the name for some reason we'll only need
# to change it in one place
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEB_SEARCH = "web_search"
