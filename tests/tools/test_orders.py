import pytest
from tools.orders import ORDER_CUSTOMER, ORDER_EMAIL, ORDER_ID, ORDER_ITEM, ORDER_STATUS, ORDER_TRACKING, load_orders_documents

#test sample taken from data/orders.json
sample_order = {
    ORDER_ID: "ORD-201",
    ORDER_ITEM: "Butter Chicken",
    ORDER_CUSTOMER: "Priya Nair",
    ORDER_STATUS: "Out for Delivery",
    ORDER_TRACKING: "SS201TRK",
    ORDER_EMAIL: "priya@example.com"
  }
test_orders_data = [
    ("ORD-201", sample_order),
    ("SS201TRK", sample_order),
    ("priya@example.com", sample_order)
    ]

@pytest.mark.parametrize("key, expected_order", test_orders_data)
def test_load_orders_documents(key, expected_order):
    file_path = 'data/orders.json'
    orders = load_orders_documents(file_path)
    assert len(orders) == 5, "Expected 5 order documents"
    order = orders[key]
    assert order == expected_order, f"Expected order for key {key} to match the sample order"
    print(f"Order for key {key}: {order}")