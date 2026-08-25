`timescale 1ns / 1ps

module uart_top_tb;

    // -----------------------------------------------------------------------
    // Signals
    // -----------------------------------------------------------------------
    reg  clk_125MHz;
    reg  reset;
    reg  rx;

    wire data_ready;
    wire rx_idle;
    wire match;
    wire full;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    uart_top dut (
        .clk_125MHz (clk_125MHz),
        .reset      (reset),
        .rx         (rx),
        .data_ready (data_ready),
        .rx_idle    (rx_idle),
        .match      (match),
        .full       (full)
    );

    // -----------------------------------------------------------------------
    // Clock: 125 MHz = 8 ns period
    // -----------------------------------------------------------------------
    initial clk_125MHz = 0;
    always #4 clk_125MHz = ~clk_125MHz;

    // -----------------------------------------------------------------------
    // Baud: 115200 ? M=68, 16x oversample ? bit period = 68*16*8 = 8704 ns
    // -----------------------------------------------------------------------
    localparam BIT_PERIOD = 8704;

    // -----------------------------------------------------------------------
    // Task: send one 8N1 UART byte (LSB first, UART standard)
    // -----------------------------------------------------------------------
    task send_byte;
        input [7:0] data;
        integer i;
        begin
            // Start bit
            rx = 1'b0;
            #(BIT_PERIOD);

            // 8 data bits LSB first
            for (i = 0; i < 8; i = i + 1) begin
                rx = data[i];
                #(BIT_PERIOD);
            end

            // Stop bit
            rx = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    // -----------------------------------------------------------------------
    // Monitor signals
    // -----------------------------------------------------------------------
    always @(posedge data_ready)
        $display("[%0t ns] data_ready pulse | rx_data=0x%02X | match=%b | full=%b",
                  $time, dut.rx_data, match, full);

    always @(posedge match)
        $display("[%0t ns] *** MATCH HIGH *** rx_data=0x%02X", $time, dut.rx_data);

    always @(posedge full)
        $display("[%0t ns] *** FIFO FULL ***", $time);

    // -----------------------------------------------------------------------
    // Stimulus
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("uart_top_tb.vcd");
        $dumpvars(0, uart_top_tb);

        // Initialize
        rx    = 1'b1;   // UART idle is HIGH
        reset = 1'b0;   // active-HIGH reset: de-asserted

        $display("================================================");
        $display("  uart_top Testbench  125MHz / 115200 baud");
        $display("================================================");

        // Apply reset (active HIGH)
        @(negedge clk_125MHz);
        reset = 1'b1;
        repeat (20) @(posedge clk_125MHz);
        reset = 1'b0;
        repeat (5)  @(posedge clk_125MHz);
        $display("[%0t ns] Reset released", $time);

        // ---------------------------------------------------------------
        // Test 1: Send 0xA5 ? match should go HIGH
        // ---------------------------------------------------------------
        $display("\n[%0t ns] TEST 1: Sending 0xA5 (expect match=1)", $time);
        send_byte(8'hA5);
        #(BIT_PERIOD);

        if (match === 1'b1)
            $display("[%0t ns] PASS - match=1 for 0xA5", $time);
        else
            $display("[%0t ns] FAIL - match=0 for 0xA5", $time);

        // ---------------------------------------------------------------
        // Test 2: Send 0x12 ? match should stay LOW
        // ---------------------------------------------------------------
        $display("\n[%0t ns] TEST 2: Sending 0x12 (expect match=0)", $time);
        send_byte(8'h12);
        #(BIT_PERIOD);

        if (match === 1'b0)
            $display("[%0t ns] PASS - match=0 for 0x12", $time);
        else
            $display("[%0t ns] FAIL - match=1 for 0x12 (should be 0)", $time);

        // ---------------------------------------------------------------
        // Test 3: Fill FIFO (depth=16) ? full should go HIGH
        // ---------------------------------------------------------------
        $display("\n[%0t ns] TEST 3: Filling FIFO (16 bytes)", $time);

        repeat (16) begin
            send_byte(8'hAA);
        end

        #(BIT_PERIOD * 2);

        if (full === 1'b1)
            $display("[%0t ns] PASS - FIFO full after 16 bytes", $time);
        else
            $display("[%0t ns] FAIL - FIFO not full", $time);

        // ---------------------------------------------------------------
        // Test 4: Send 0xA5 again after filling FIFO
        // ---------------------------------------------------------------
        $display("\n[%0t ns] TEST 4: Sending 0xA5 again", $time);
        send_byte(8'hA5);
        #(BIT_PERIOD * 2);

        if (match === 1'b1)
            $display("[%0t ns] PASS - match=1 again for 0xA5", $time);
        else
            $display("[%0t ns] FAIL - match=0", $time);

        // ---------------------------------------------------------------
        // Check rx_idle
        // ---------------------------------------------------------------
        #(BIT_PERIOD * 5);
        if (rx_idle === 1'b1)
            $display("[%0t ns] PASS - rx_idle=1 (receiver back to idle)", $time);
        else
            $display("[%0t ns] FAIL - rx_idle=0 (receiver stuck)", $time);

        $display("\n================================================");
        $display("  Simulation complete at %0t ns", $time);
        $display("================================================");
        $finish;
    end

    // -----------------------------------------------------------------------
    // Watchdog: 100 ms timeout
    // -----------------------------------------------------------------------
    initial begin
        #100_000_000;
        $display("[%0t ns] WATCHDOG timeout - simulation took too long", $time);
        $finish;
    end

endmodule