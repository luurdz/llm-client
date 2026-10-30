---
category: sistemas
---

# Manual del sistema de reservas Brújula

Rumbo Sur Viajes · Documento interno · Versión 2026.2



## 1. Qué es Brújula

Brújula es el sistema interno que usan los vendedores de Rumbo Sur Viajes para crear, modificar y cancelar reservas. Se conecta con las aerolíneas, los hoteles y la pasarela de pagos, y guarda el historial completo de cada reserva. Se accede desde el navegador, en brujula.rumbosur.example, con el usuario y la contraseña corporativos.

Cada reserva tiene un **localizador** de seis caracteres, por ejemplo `RSV7K2`. El localizador es el dato que hay que pedirle siempre al cliente para buscar su reserva; el nombre del pasajero no alcanza, porque puede haber homónimos.

## 2. Estados de una reserva

Una reserva pasa por estos estados, en este orden:

- **PENDIENTE:** el vendedor cargó los datos, pero el cliente todavía no pagó la seña. Las reservas pendientes se cancelan solas a las 48 horas.
- **SEÑADA:** el cliente pagó la seña del 30%. Los cupos de vuelo y hotel quedan bloqueados.
- **CONFIRMADA:** el cliente pagó el saldo y los proveedores confirmaron los servicios. Recién en este estado se pueden emitir los vouchers.
- **EMITIDA:** los vouchers y los pasajes ya fueron enviados al cliente.
- **CERRADA:** el viaje terminó. Las reservas cerradas no se pueden modificar.

Una reserva puede pasar a **CANCELADA** desde cualquier estado anterior a EMITIDA. Para cancelar una reserva EMITIDA hace falta la autorización de un supervisor.



Rumbo Sur Viajes · Documento interno · Versión 2026.2



## 3. Códigos de error

Cuando una operación falla, Brújula muestra un código de error. Estos son los más frecuentes y qué hacer en cada caso:

- **RS-1001 — Localizador inexistente:** el localizador no existe o tiene un error de tipeo. Verificar con el cliente; recordar que el sistema no usa las letras I ni O, para no confundirlas con los números 1 y 0.
- **RS-1042 — Pago rechazado:** la pasarela de pagos rechazó la tarjeta. No reintentar más de dos veces con la misma tarjeta, porque el banco puede bloquearla. Pedir otro medio de pago.
- **RS-1050 — Cupo agotado:** el hotel o el vuelo ya no tienen lugar para esa fecha. Ofrecer fechas alternativas desde la pestaña "Disponibilidad".
- **RS-2003 — Tarifa vencida:** el precio que se cotizó ya no está vigente. Volver a cotizar; las tarifas de vuelos se mantienen solo por 30 minutos.
- **RS-2210 — Documento inválido:** el número de pasaporte o de DNI no tiene el formato correcto para el destino. Revisar que el documento esté vigente y que se haya elegido el tipo correcto.
- **RS-3100 — Proveedor sin respuesta:** el sistema de la aerolínea o del hotel no respondió a tiempo. Esperar cinco minutos y reintentar. Si el error se repite tres veces, abrir un ticket a la mesa de ayuda.

## 4. Mesa de ayuda

Los problemas técnicos que el vendedor no puede resolver se informan a la mesa de ayuda de sistemas, abriendo un ticket desde el menú "Soporte" de Brújula. Cada ticket necesita el localizador, el código de error y una captura de pantalla.

Los tickets tienen tres prioridades:

- **P1:** el sistema no funciona para nadie. Se atiende en menos de 30 minutos, también los fines de semana.
- **P2:** un error impide confirmar o emitir una reserva concreta. Se atiende en menos de 4 horas hábiles.
- **P3:** consultas y problemas menores. Se atienden en menos de 2 días hábiles.

## 5. Buenas prácticas

- No compartir el usuario de Brújula: cada operación queda registrada con el nombre del vendedor que la hizo.
- Cerrar la sesión al terminar el turno. Las sesiones abiertas se cierran solas después de 20 minutos sin actividad.
- Antes de emitir, revisar siempre que el nombre del pasajero coincida exactamente con su documento. Un error en el nombre puede obligar a reemitir el pasaje, con un costo de la aerolínea.



Rumbo Sur Viajes · Documento interno · Versión 2026.2
