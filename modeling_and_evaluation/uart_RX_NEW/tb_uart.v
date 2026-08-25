`timescale 1ns / 1ps

module uart_top_tb();

    parameter CLK_PERIOD = 8;
    parameter BAUD_RATE  = 115200;
    parameter BIT_PERIOD = 1_000_000_000 / BAUD_RATE;

    reg        clk;
    reg        reset;
    reg        rx;
    reg        read_from_fifo;

    wire [7:0] data_out;
    wire       fifo_empty;
    wire       fifo_full;

    uart_top
        #(
            .DBITS         (8),
            .SB_TICK       (16),
            .BR_LIMIT      (68),
            .BR_BITS       (7),
            .FIFO_ADDR_BITS(4)
        )
        dut
        (
            .clk           (clk),
            .reset         (reset),
            .rx            (rx),
            .read_from_fifo(read_from_fifo),
            .data_out      (data_out),
            .fifo_empty    (fifo_empty),
            .fifo_full     (fifo_full)
        );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    task uart_send_byte;
        input [7:0] data;
        integer i;
        begin
            rx = 1'b0;
            #(BIT_PERIOD);
            for(i = 0; i < 8; i = i + 1) begin
                rx = data[i];
                #(BIT_PERIOD);
            end
            rx = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    task read_fifo;
        begin
            @(posedge clk);
            read_from_fifo = 1'b1;
            @(posedge clk);
            read_from_fifo = 1'b0;
            @(posedge clk);
            $display("TIME: %0t | FIFO Read: 0x%0h (%0d) | empty=%b full=%b",
                      $time, data_out, data_out, fifo_empty, fifo_full);
        end
    endtask

    initial begin
        reset          = 1'b1;
        rx             = 1'b1;    // <-- MUST be HIGH (idle) from the start
        read_from_fifo = 1'b0;

        repeat(10) @(posedge clk);
        reset = 1'b0;
        repeat(5) @(posedge clk); // extra settling time after reset

        $display("==============================================");
        $display("     UART RX TESTBENCH - 125MHz / 115200     ");
        $display("==============================================");

        $display("\n[TEST 1] Sending 0x41 (A)");
        uart_send_byte(8'h41);
        #(BIT_PERIOD * 3);
        read_fifo();

        $display("\n[TEST 2] Sending 0x42 (B)");
        uart_send_byte(8'h42);
        #(BIT_PERIOD * 3);
        read_fifo();

        $display("\n[TEST 3] Sending 5 bytes without reading...");
        uart_send_byte(8'h01);
        uart_send_byte(8'h02);
        uart_send_byte(8'h03);
        uart_send_byte(8'h04);
        uart_send_byte(8'h05);
        #(BIT_PERIOD * 3);
        $display("FIFO Status -> empty=%b | full=%b", fifo_empty, fifo_full);
        repeat(5) read_fifo();

        $display("\n[TEST 4] Sending 0xFF");
        uart_send_byte(8'hFF);
        #(BIT_PERIOD * 3);
        read_fifo();

        $display("\n[TEST 4] Sending 0x00");
        uart_send_byte(8'h00);
        #(BIT_PERIOD * 3);
        read_fifo();

        $display("\n==============================================");
        $display("           TESTBENCH COMPLETE                ");
        $display("==============================================\n");
        $finish;
    end

    initial begin
        $monitor("TIME: %0t | rx=%b | data_ready=%b | data_out=0x%0h | empty=%b | full=%b",
                  $time, rx, dut.data_ready, data_out, fifo_empty, fifo_full);
    end

    // Watchdog - 5ms is enough for all tests
    initial begin
        #5_000_000;
        $display("TIMEOUT");
        $finish;
    end

endmodule