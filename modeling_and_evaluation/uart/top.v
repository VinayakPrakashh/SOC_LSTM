
`timescale 1ns / 1ps

module uart_top
    (
        input clk_125MHz,           // 125 MHz system clock
        input reset,                // reset button (active high)
        input rx,                   // UART receive line
        output data_ready,          // data ready flag
        output rx_idle      ,        // receiver idle indicator,
        output match,                 // indicates if received data matches expected value
        output full
    );
    
    // Internal signals
    wire sample_tick;
    wire [7:0] rx_data;
    wire rx_done;
    
    // Baud rate generator instance
    baud_rate_generator #(
        .N(7),
        .M(68)                      // 125MHz / (115200 * 16) ≈ 68
    ) baud_gen (
        .clk_125MHz(clk_125MHz),
        .reset(reset),
        .tick(sample_tick)
    );

    
    // UART receiver instance
    uart_receiver #(
        .DBITS(8),
        .SB_TICK(16)
    ) uart_rx (
        .clk_125MHz(clk_125MHz),
        .reset(reset),
        .rx(rx),
        .sample_tick(sample_tick),
        .data_ready(rx_done),
        .data_out(rx_data)
    );

    assign match = (rx_data == 8'hA5);  // Example expected value for verification

    assign data_ready = rx_done;
    assign rx_idle = (uart_rx.state == 2'b00);  // Monitor idle state
    fifo #(
        .DATA_SIZE(8),
        .ADDR_SPACE_EXP(4)
    ) uart_fifo (
        .clk(clk_125MHz),
        .reset(reset),
        .write_to_fifo(rx_done),
        .read_from_fifo(1'b0),  // No reading in this example
        .write_data_in(rx_data),
        .read_data_out(),       // Not used in this example
        .empty(),              // Not used in this example
        .full(full)                // Not used in this example
    );
    
endmodule
