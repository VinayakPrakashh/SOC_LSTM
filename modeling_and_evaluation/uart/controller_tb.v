`timescale 1ns / 1ps

module tb_fifo_input_system;

    // Parameters
    parameter DATA_SIZE = 8;
    parameter ADDR_SPACE_EXP = 4;
    parameter OUTPUT_DATA_WIDTH = 80;
    parameter ADDR_WIDTH = 5;
    parameter CLK_PERIOD = 10; // 100MHz clock
    
    // Testbench signals
    reg clk;
    reg rst_n;
    reg start;
    
    wire [OUTPUT_DATA_WIDTH-1:0] output_data;
    wire done;
    wire wr_en;
    wire fifo_empty;
    wire fifo_full;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    fifo_input_system #(
        .DATA_SIZE(DATA_SIZE),
        .ADDR_SPACE_EXP(ADDR_SPACE_EXP),
        .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .output_data(output_data),
        .done(done),
        .wr_en(wr_en),
        .fifo_empty(fifo_empty),
        .fifo_full(fifo_full)
    );
    
    // Test sequence
    initial begin
        // Initialize signals
        rst_n = 0;
        start = 0;
        
        // Create waveform dump
        $dumpfile("fifo_input_system.vcd");
        $dumpvars(0, tb_fifo_input_system);
        
        // Display header
        $display("==================================================");
        $display("   FIFO Input System Testbench");
        $display("==================================================");
        $display("Time\t\tReset\tStart\tEmpty\tFull\tDone\tWrEn\tOutput Data");
        $display("--------------------------------------------------");
        
        // Apply reset
        #20;
        rst_n = 1;
        #20;
        
        $display("%0t ns:\tReset released, FIFO pre-filled with 10 values", $time);
        $display("%0t ns:\tFIFO Status - Empty: %b, Full: %b", $time, fifo_empty, fifo_full);
        
        // Wait a few cycles
        #40;
        
        // Assert start signal
        $display("\n%0t ns:\tAsserting START signal", $time);
        start = 1;
        #10;
        start = 0;
        
        // Wait for processing to complete
        wait(done == 1);
        #10;
        
        $display("\n%0t ns:\tProcessing COMPLETE", $time);
        $display("         Done: %b, Write Enable: %b", done, wr_en);
        $display("         Output Data (80-bit): 0x%h", output_data);
        $display("\n         Expected: 0x0A_09_08_07_06_05_04_03_02_01");
        $display("         Byte breakdown:");
        $display("           Bytes [79:72] = 0x%h (expected 0x0A)", output_data[79:72]);
        $display("           Bytes [71:64] = 0x%h (expected 0x09)", output_data[71:64]);
        $display("           Bytes [63:56] = 0x%h (expected 0x08)", output_data[63:56]);
        $display("           Bytes [55:48] = 0x%h (expected 0x07)", output_data[55:48]);
        $display("           Bytes [47:40] = 0x%h (expected 0x06)", output_data[47:40]);
        $display("           Bytes [39:32] = 0x%h (expected 0x05)", output_data[39:32]);
        $display("           Bytes [31:24] = 0x%h (expected 0x04)", output_data[31:24]);
        $display("           Bytes [23:16] = 0x%h (expected 0x03)", output_data[23:16]);
        $display("           Bytes [15:8]  = 0x%h (expected 0x02)", output_data[15:8]);
        $display("           Bytes [7:0]   = 0x%h (expected 0x01)", output_data[7:0]);
        
        $display("\n%0t ns:\tFIFO Status after processing - Empty: %b, Full: %b", $time, fifo_empty, fifo_full);
        
        // Wait a few more cycles
        #50;
        
        // Test second start pulse
        $display("\n%0t ns:\tTesting second START pulse (FIFO should be empty)", $time);
        start = 1;
        #10;
        start = 0;
        
        #100;
        
        // Verify output data matches expected pattern
        if (output_data == 80'h0A_09_08_07_06_05_04_03_02_01) begin
            $display("\n==================================================");
            $display("   TEST PASSED!");
            $display("==================================================");
        end else begin
            $display("\n==================================================");
            $display("   TEST FAILED!");
            $display("   Expected: 0x0A_09_08_07_06_05_04_03_02_01");
            $display("   Got:      0x%h", output_data);
            $display("==================================================");
        end
        
        // End simulation
        #50;
        $display("\nSimulation finished at %0t ns", $time);
        $finish;
    end
    


endmodule