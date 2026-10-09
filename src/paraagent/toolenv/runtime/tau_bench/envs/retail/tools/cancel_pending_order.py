# Copyright Sierra

import json
from typing import Any, Dict
from tau_bench.envs.tool import Tool

class CancelPendingOrder(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str, reason: str) -> str:
        
        orders = data["orders"]
        if order_id not in orders:
            return "Error: order not found"
        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be cancelled"

        if reason not in ["no longer needed", "ordered by mistake"]:
            return "Error: invalid reason"

        refunds = []
        for payment in order["payment_history"]:
            payment_id = payment["payment_method_id"]
            refund = {
                "transaction_type": "refund",
                "amount": payment["amount"],
                "payment_method_id": payment_id,
            }
            refunds.append(refund)
            if "gift_card" in payment_id:  
                payment_method = data["users"][order["user_id"]]["payment_methods"][
                    payment_id
                ]
                payment_method["balance"] += payment["amount"]
                payment_method["balance"] = round(payment_method["balance"], 2)

        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)

        return json.dumps(order)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "description": (
                    "Cancel an order that is still pending. Orders that have already been processed "
                    "or delivered cannot be cancelled. A successful cancellation changes the order "
                    "status to 'cancelled' and refunds the payment. "
                    "The refund will be added to the user's gift card balance immediately if the payment "
                    "was made using a gift card, otherwise the refund would take 5-7 business days to process. "
                    "The function returns the order details after the cancellation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["no longer needed", "ordered by mistake"],
                            "description": "The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.",
                        },
                    },
                    "required": ["order_id", "reason"],
                },
            },
        }
