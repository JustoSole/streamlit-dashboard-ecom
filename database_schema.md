# Database Schema

This document outlines the schema for the tables used in the analytics dashboard, as sourced from BigQuery.

---

## `RAW_SHOPIFY_ORDERS`

| Field Name                       | Type    | Mode     | Description                                                              |
| -------------------------------- | ------- | -------- | ------------------------------------------------------------------------ |
| `ORDER_ID`                       | STRING  | REQUIRED | PK - Identificador único global de la orden en Shopify en formato GID.   |
| `ORDER_NUMBER`                   | STRING  | NULLABLE | Número visible de la orden en Shopify, precedido por '#'.                |
| `ORDER_CREATE_DATE`              | TIMESTAMP | NULLABLE | Timestamp de creación de la orden en la tienda.                          |
| `ORDER_FULFILLED_DATE`           | TIMESTAMP | NULLABLE | Timestamp en que la orden fue marcada como completada por el sistema.    |
| `ORDER_CANCEL_DATE`              | TIMESTAMP | NULLABLE | Timestamp en que la orden fue cancelada (si aplica).                     |
| `ORDER_CANCEL_REASON`            | STRING  | NULLABLE | Motivo registrado para la cancelación de la orden.                       |
| `ORDER_CANCEL_NOTE`              | STRING  | NULLABLE | Nota adicional o comentario asociado a la cancelación de la orden.       |
| `ORDER_RETURN_STATUS`            | STRING  | NULLABLE | Indica si la orden fue devuelta o si no aplica devolución.               |
| `ORDER_RISK_RECOMENDATION`       | STRING  | NULLABLE | Clasificación de riesgo asignada por Shopify.                            |
| `ORDER_ORIGINAL_LINEITEM_QTY`    | INTEGER | NULLABLE | Cantidad original total de ítems en la orden.                            |
| `ORDER_FINAL_LINEITEM_QTY`       | INTEGER | NULLABLE | Cantidad final de ítems en la orden (después de devoluciones o cambios).  |
| `ORDER_ORIGINAL_SUBTOTAL_AMOUNT` | FLOAT   | NULLABLE | Subtotal original de la orden antes de impuestos y descuentos.           |
| `ORDER_FINAL_SUBTOTAL_AMOUNT`    | FLOAT   | NULLABLE | Subtotal final de la orden después de ajustes.                           |
| `ORDER_ORIGINAL_SHIPPING_AMOUNT` | FLOAT   | NULLABLE | Costo de envío original calculado al momento de la compra.               |
| `ORDER_FINAL_SHIPPING_AMOUNT`    | FLOAT   | NULLABLE | Costo de envío final tras modificaciones.                                |
| `ORDER_ORIGINAL_TAX_AMOUNT`      | FLOAT   | NULLABLE | Monto original de impuestos aplicados en la orden.                       |
| `ORDER_FINAL_TAX_AMOUNT`         | FLOAT   | NULLABLE | Monto final de impuestos tras ajustes.                                   |
| `ORDER_ORIGINAL_DISCOUNT_AMOUNT` | FLOAT   | NULLABLE | Total de descuentos aplicados inicialmente en la orden.                  |
| `ORDER_FINAL_DISCOUNT_AMOUNT`    | FLOAT   | NULLABLE | Total de descuentos finales tras ajustes.                                |
| `ORDER_ORIGINAL_TOTAL_AMOUNT`    | FLOAT   | NULLABLE | Monto total original de la orden.                                        |
| `ORDER_FINAL_TOTAL_AMOUNT`       | FLOAT   | NULLABLE | Monto total final de la orden después de todos los ajustes.              |
| `ORDER_FINAL_REFOUND_AMOUNT`     | FLOAT   | NULLABLE | Monto total reembolsado al cliente.                                      |
| `ORDER_FINAL_NET_AMOUNT`         | FLOAT   | NULLABLE | Monto neto percibido tras descuentos y reembolsos.                       |
| `ORDER_DISCOUNT_CODE`            | STRING  | NULLABLE | Código promocional o cupón aplicado en la orden.                         |
| `ORDER_FINANCIAL_STATUS`         | STRING  | NULLABLE | Estado financiero de la orden (ej: PAID, REFUNDED).                      |
| `ORDER_FULFILLMENT_STATUS`       | STRING  | NULLABLE | Estado logístico de cumplimiento de la orden.                            |
| `CUSTOMER_ID`                    | STRING  | NULLABLE | Identificador del cliente que realizó la orden (formato GID).            |
| `SHIPPING_ADDRESS`               | STRING  | NULLABLE | Dirección completa de destino del envío.                                 |
| `SHIPPING_CITY`                  | STRING  | NULLABLE | Ciudad de destino del envío.                                             |
| `SHIPPING_STATE`                 | STRING  | NULLABLE | Provincia o estado de destino del envío.                                 |
| `SHIPPING_ZIPCODE`               | STRING  | NULLABLE | Código postal de destino del envío.                                      |
| `SHIPPING_LATITUDE`              | FLOAT   | NULLABLE | Latitud geográfica de la dirección de envío.                             |
| `SHIPPING_LONGITUDE`             | FLOAT   | NULLABLE | Longitud geográfica de la dirección de envío.                            |
| `SHIPPING_SERVICE_LEVEL`         | STRING  | NULLABLE | Nivel de servicio de envío seleccionado (ej: standard, express).         |
| `SHIPPING_SOURCE`                | STRING  | NULLABLE | Origen o canal logístico utilizado (ej: shopify, collective).            |
| `ORDER_FIRST_VISIT_SOURCE`       | STRING  | NULLABLE | Fuente de tráfico de la primera visita del cliente (ej: Google).         |
| `ORDER_FIRST_VISIT_SOURCE_TYPE`  | STRING  | NULLABLE | Tipo de canal de la primera visita (ej: SEO, direct).                    |
| `ORDER_FIRST_VISIT_DATE`         | TIMESTAMP | NULLABLE | Timestamp de la primera visita del cliente antes de comprar.             |
| `ORDER_MOMENTS_QTY`              | INTEGER | NULLABLE | Número de interacciones clave registradas para la orden.                 |
| `ORDER_DAYS_TO_CONVERSION`       | FLOAT   | NULLABLE | Cantidad de días entre la primera visita y la compra.                    |
| `ORDER_DELIVERY_DATE`            | TIMESTAMP | NULLABLE | Fecha de entrega efectiva registrada para la orden.                      |
| `ORDER_ESTIMATED_DELIVERY_DATE`  | TIMESTAMP | NULLABLE | Fecha de entrega estimada informada al cliente.                          |
| `ORDER_DELIVERED_ON_TIME`        | STRING  | NULLABLE | Indica si la orden fue entregada dentro del plazo estimado.              |
| `ORDER_TAGS`                     | STRING  | NULLABLE | Etiquetas internas aplicadas a la orden en Shopify.                      |

---

## `RAW_SHOPIFY_CUSTOMERS`

| Field Name                        | Type      | Mode     | Description                                                                 |
| --------------------------------- | --------- | -------- | --------------------------------------------------------------------------- |
| `CUSTOMER_ID`                     | STRING    | REQUIRED | PK - Identificador único del cliente en Shopify (formato GID).              |
| `CUSTOMER_CREATE_DATE`            | TIMESTAMP | NULLABLE | Timestamp de creación del perfil del cliente en Shopify.                    |
| `CUSTOMER_DISPLAY_NAME`           | STRING    | NULLABLE | Nombre completo del cliente como aparece en Shopify.                        |
| `CUSTOMER_EMAIL`                  | STRING    | NULLABLE | Correo electrónico del cliente asociado a su cuenta.                         |
| `CUSTOMER_PHONE`                  | FLOAT     | NULLABLE | Teléfono de contacto del cliente asociado a su cuenta.                       |
| `CUSTOMER_CITY`                   | STRING    | NULLABLE | Ciudad registrada del cliente, no necesariamente es su direccion de envio.   |
| `CUSTOMER_STATE`                  | STRING    | NULLABLE | Provincia o estado de residencia del cliente, no necesariamente es su direccion de envio. |
| `CUSTOMER_COUNTRY`                | STRING    | NULLABLE | País del cliente, no necesariamente es su direccion de envio.                |
| `CUSTOMER_TOTAL_SPENT_AMOUNT`     | FLOAT     | NULLABLE | Monto total gastado por el cliente en todas sus compras.                    |
| `CUSTOMER_TOTAL_ORDERS`           | INTEGER   | NULLABLE | Cantidad total de órdenes realizadas por el cliente.                        |
| `CUSTOMER_LAST_ORDER_ID`          | STRING    | NULLABLE | ID de la última orden realizada por el cliente (formato GID).               |
| `CUSTOMER_LAST_ORDER_GA_TRANSACTION_ID` | FLOAT | NULLABLE | ID de transacción de Google Analytics de la última orden. |
| `CUSTOMER_LAST_ORDER_ORDER_NUMBER`| STRING    | NULLABLE | Número de orden de la ultima orden del cliente.                             |
| `ORDER_CREATE_DATE`               | TIMESTAMP | NULLABLE | Timestamp de creación de la última orden del cliente.                       |

---

## `RAW_SHOPIFY_PRODUCTS`

| Field Name                 | Type    | Mode     | Description                                                          |
| -------------------------- | ------- | -------- | -------------------------------------------------------------------- |
| `VARIANT_ID`               | STRING  | REQUIRED | PK - Identificador único del variante del producto en Shopify (formato GID). |
| `PRODUCT_ID`               | STRING  | NULLABLE | Identificador único del producto principal en Shopify (formato GID).   |
| `VARIANT_EAN`              | STRING  | NULLABLE | Código EAN asociado al variante del producto (código de barras).     |
| `VARIANT_SKU`              | STRING  | NULLABLE | SKU (Stock Keeping Unit) del variante del producto.                  |
| `VARIANT_CREATE_DATE`      | TIMESTAMP | NULLABLE | Timestamp de creación del producto o variante.                       |
| `VARIANT_LAST_UPDATE_DATE` | TIMESTAMP | NULLABLE | Timestamp de última actualización del producto o variante.           |
| `PRODUCT_STATUS`           | STRING  | NULLABLE | Estado actual del producto en Shopify (ej: ACTIVE, ARCHIVED).        |
| `PRODUCT_CATEGORY`         | STRING  | NULLABLE | Categoría comercial asignada al producto.                            |
| `PRODUCT_TYPE`             | STRING  | NULLABLE | Tipo de producto definido en Shopify.                                |
| `PRODUCT_VENDOR`           | STRING  | NULLABLE | Marca o proveedor que vende el producto.                             |
| `PRODUCT_NAME`             | STRING  | NULLABLE | Nombre comercial del producto.                                       |
| `PRODUCT_DESCRIPTION`      | STRING  | NULLABLE | Descripción larga del producto para clientes.                        |
| `PRODUCT_PRICE`            | FLOAT   | NULLABLE | Precio de venta del producto.                                        |
| `PRODUCT_COMPARE_AT_PRICE` | FLOAT   | NULLABLE | Precio original para comparación (sin descuento).                    |
| `VARTIANT_STOCK_QTY`       | INTEGER | NULLABLE | Cantidad de unidades disponibles en inventario para este variante.   |
| `PRODUCT_TAGS`             | STRING  | NULLABLE | Etiquetas comerciales o de clasificación del producto.               |
| `PRODUCT_IS_COLLECTIVE`    | BOOLEAN | NULLABLE | Indica si el producto pertenece a Shopify Collective.                |
| `PRODUCT_HAS_PRICE_MARKDOWN`| BOOLEAN | NULLABLE | Indica si el precio está actualmente rebajado respecto al precio original. |

---

## `RAW_SHOPIFY_ORDERS_LINEITEMS`

| Field Name                      | Type    | Mode     | Description                                                                 |
| ------------------------------- | ------- | -------- | --------------------------------------------------------------------------- |
| `LINEITEM_ID`                   | STRING  | REQUIRED | PK - Identificador único del ítem de línea en Shopify (formato GID).        |
| `ORDER_CREATE_DATE`             | TIMESTAMP | NULLABLE | Fecha y hora de creación de la orden.                                       |
| `ORDER_ID`                      | STRING  | NULLABLE | Identificador único de la orden (formato GID en Shopify).                   |
| `ORDER_NUMBER`                  | STRING  | NULLABLE | Número de orden visible para el cliente.                                    |
| `ORDER_FULFILLMENT_STATUS`      | STRING  | NULLABLE | Estado de cumplimiento del pedido (ej. FULFILLED, PARTIAL, etc.).           |
| `ORDER_FINANCIAL_STATUS`        | STRING  | NULLABLE | Estado financiero de la orden (ej. PAID, PENDING, etc.).                    |
| `CUSTOMER_ID`                   | STRING  | NULLABLE | Identificador único del cliente (formato GID).                              |
| `PRODUCT_ID`                    | STRING  | NULLABLE | Identificador del producto en Shopify (formato GID).                        |
| `PRODUCT_SKU`                   | STRING  | NULLABLE | Código SKU del producto.                                                    |
| `PRODUCT_NAME`                  | STRING  | NULLABLE | Nombre del producto.                                                        |
| `PRODUCT_VENDOR`                | STRING  | NULLABLE | Marca o proveedor del producto.                                             |
| `PRODUCT_TYPE`                  | STRING  | NULLABLE | Categoría del producto (ej. Sprays, Balls, etc.).                           |
| `LINEITEM_ORIGINAL_QTY`         | INTEGER | NULLABLE | Cantidad original ordenada del ítem.                                        |
| `LINEITEM_FINAL_QTY`            | INTEGER | NULLABLE | Cantidad final procesada del ítem (puede reflejar cambios posteriores).     |
| `LINEITEM_UNIT_ORIGINAL_AMOUNT` | FLOAT   | NULLABLE | Precio unitario original del ítem (antes de descuentos o modificaciones).   |
| `LINEITEM_UNIT_FINAL_AMOUNT`    | FLOAT   | NULLABLE | Precio unitario final del ítem después de ajustes o descuentos.             |

---

## `RAW_GA_CAMPAIGN_METRICS`

| Field Name                        | Type    | Mode     | Description                                             |
| --------------------------------- | ------- | -------- | ------------------------------------------------------- |
| `SK_DATECAMPAIGN`                 | STRING  | REQUIRED | PK: Surrogate key formed by CAMPAIGN_DATE + CAMPAIGN_ID |
| `CAMPAIGN_DATE`                   | TIMESTAMP | NULLABLE | Date of the campaign                                    |
| `CAMPAIGN_ID`                     | STRING  | NULLABLE | Original ID of the campaign                             |
| `CAMPAIGN_NAME`                   | STRING  | NULLABLE | Name of the campaign                                    |
| `CAMPAIGN_IMPRESSIONS`            | FLOAT   | NULLABLE | Number of ad impressions                                |
| `CAMPAIGN_CLICKS`                 | FLOAT   | NULLABLE | Number of ad clicks                                     |
| `CAMPAIGN_SESSIONS`               | FLOAT   | NULLABLE | Number of sessions started from the campaign            |
| `CAMPAIGN_USERS`                  | FLOAT   | NULLABLE | Total users attributed to the campaign                  |
| `CAMPAIGN_BOUNCE_RATE`            | FLOAT   | NULLABLE | Bounce rate for the campaign                            |
| `CAMPAIGN_AD_REVENUE`             | FLOAT   | NULLABLE | Ad revenue attributed to the campaign                   |
| `CAMPAIGN_TOTAL_REVENUE`          | FLOAT   | NULLABLE | Total revenue generated by the campaign                 |
| `CAMPAIGN_COST`                   | FLOAT   | NULLABLE | Cost of the campaign                                    |
| `CAMPAIGN_ROAS`                   | FLOAT   | NULLABLE | Return on ad spend (ROAS)                               |
| `CAMPAIGN_NEW_USERS`              | FLOAT   | NULLABLE | New users acquired through the campaign                 |
| `CAMPAIGN_FIRST_TIME_PURCHARSERS` | FLOAT   | NULLABLE | Users who made their first purchase                     |
| `CAMPAIGN_AVG_SESSION_DURATION`   | FLOAT   | NULLABLE | Average session duration (seconds)                      |
| `CAMPAIGN_ADD_TO_CART_EVENTS`     | FLOAT   | NULLABLE | Number of add-to-cart events                            |
| `CAMPAIGN_BEGIN_CHECKOUT_EVENTS`  | FLOAT   | NULLABLE | Number of begin checkout events                         |
| `CAMPAIGN_PURCHASE_EVENTS`        | FLOAT   | NULLABLE | Number of completed purchases                           |

---

## `RAW_GA_CAMPAIGN_TRANSACTIONS`

| Field Name                   | Type    | Mode     | Description                                                    |
| ---------------------------- | ------- | -------- | -------------------------------------------------------------- |
| `SK_ROWNUMBER`               | STRING  | REQUIRED | PK: Surrogate key given by the row number                      |
| `CAMPAIGN_DATE`              | TIMESTAMP | NULLABLE | Date when the campaign was active                              |
| `TRANSACTION_ID`             | STRING  | NULLABLE | Transaction identifier if applicable                           |
| `CAMPAIGN_SOURCE`            | STRING  | NULLABLE | Source of the campaign (e.g., direct, referral, paid)          |
| `CAMPAIGN_PRIMARY_GROUP`     | STRING  | NULLABLE | Primary campaign grouping category                             |
| `CAMPAIGN_SOURCE_PLATFORM`   | STRING  | NULLABLE | Platform where the campaign ran (e.g., Google, Facebook)       |
| `CAMPAIGN_ID`                | STRING  | NULLABLE | Original campaign identifier from the source                   |
| `CAMPAIGN_SESSIONS`          | FLOAT   | NULLABLE | Number of sessions attributed to the campaign                  |
| `CAMPAIGN_USERS`             | FLOAT   | NULLABLE | Number of users involved in the campaign                       |
| `CAMPAIGN_AVG_SESSION_DURATION` | FLOAT   | NULLABLE | Average session duration for the campaign (in seconds)       |
| `CAMPAIGN_LANDING_PAGE`      | STRING  | NULLABLE | First landing page accessed via the campaign                   | 