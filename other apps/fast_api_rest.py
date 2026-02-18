from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI GET, POST & PUT")

# -----------------------------
# Fake Database (In-Memory)
# -----------------------------
items_db = {}

# -----------------------------
# Pydantic Model
# -----------------------------
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True


# -----------------------------
# POST - Create Item
# -----------------------------
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    items_db[item_id] = item
    return {
        "message": "Item created successfully",
        "item_id": item_id,
        "item": item
    }


# -----------------------------
# GET - Get Single Item
# -----------------------------
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        return {"error": "Item not found"}

    return {
        "item_id": item_id,
        "item": items_db[item_id]
    }


# -----------------------------
# GET - Get All Items
# -----------------------------
@app.get("/items")
def get_all_items():
    return items_db


# -----------------------------
# PUT - Update Item
# -----------------------------
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        return {"error": "Item not found"}

    items_db[item_id] = item
    return {
        "message": "Item updated successfully",
        "item_id": item_id,
        "item": item
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)