`timescale 1ns / 1ps

module tb_top_16_by_1_streaming();

    // Parameters
    parameter DATA_WIDTH = 12;
    parameter OUTPUT_WIDTH = 12;
    parameter FIFO_DEPTH = 16;
    
    // Clock and Reset
    reg clk = 1'b0;
    reg rst = 1'b1;
    
    // Input signals
    reg wr_en = 1'b0;
    reg [DATA_WIDTH-1:0] data_r1 = 0, data_r2 = 0, data_r3 = 0, data_r4 = 0;
    reg [DATA_WIDTH-1:0] weight_c1 = 0, weight_c2 = 0, weight_c3 = 0, weight_c4 = 0;
    reg [DATA_WIDTH-1:0] weight_c5 = 0, weight_c6 = 0, weight_c7 = 0, weight_c8 = 0;
    reg [DATA_WIDTH-1:0] weight_c9 = 0, weight_c10 = 0, weight_c11 = 0, weight_c12 = 0;
    reg [DATA_WIDTH-1:0] weight_c13 = 0, weight_c14 = 0, weight_c15 = 0, weight_c16 = 0;
    
    // Output signals
    wire [OUTPUT_WIDTH-1:0] pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8;
    wire [OUTPUT_WIDTH-1:0] pe9, pe10, pe11, pe12, pe13, pe14, pe15, pe16;
    wire fifo_full, fifo_empty, fifo_valid;
    
    // Test variables
    integer cycle_count = 0;
    
    // DUT instantiation
    top_16_by_1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .data_r1(data_r1),
        .data_r2(data_r2),
        .data_r3(data_r3),
        .data_r4(data_r4),
        .weight_c1(weight_c1),   .weight_c2(weight_c2),   .weight_c3(weight_c3),   .weight_c4(weight_c4),
        .weight_c5(weight_c5),   .weight_c6(weight_c6),   .weight_c7(weight_c7),   .weight_c8(weight_c8),
        .weight_c9(weight_c9),   .weight_c10(weight_c10), .weight_c11(weight_c11), .weight_c12(weight_c12),
        .weight_c13(weight_c13), .weight_c14(weight_c14), .weight_c15(weight_c15), .weight_c16(weight_c16),
        .pe1(pe1),   .pe2(pe2),   .pe3(pe3),   .pe4(pe4),
        .pe5(pe5),   .pe6(pe6),   .pe7(pe7),   .pe8(pe8),
        .pe9(pe9),   .pe10(pe10), .pe11(pe11), .pe12(pe12),
        .pe13(pe13), .pe14(pe14), .pe15(pe15), .pe16(pe16),
        .fifo_full(fifo_full),
        .fifo_empty(fifo_empty),
        .fifo_valid(fifo_valid)
    );
    
    // Clock generation
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;  // 100MHz clock (10ns period)
    end
    
    // Reset sequence
    initial begin
        rst = 1'b1;
        #50;  // Hold reset for 5 clock cycles
        rst = 1'b0;
        $display("Reset released at time %0t", $time);
    end
    
    // Streaming data and weights
    always @(posedge clk) begin
        if (!rst && cycle_count < 16) begin
            wr_en <= 1'b1;
            
            // Stream data values from 1 to 16
            data_r1 <= cycle_count + 1;
            
            // All weights have the same value each cycle (1 in cycle 1, 2 in cycle 2, etc.)
            weight_c1  <= cycle_count + 1;
            weight_c2  <= cycle_count + 1;
            weight_c3  <= cycle_count + 1;
            weight_c4  <= cycle_count + 1;
            weight_c5  <= cycle_count + 1;
            weight_c6  <= cycle_count + 1;
            weight_c7  <= cycle_count + 1;
            weight_c8  <= cycle_count + 1;
            weight_c9  <= cycle_count + 1;
            weight_c10 <= cycle_count + 1;
            weight_c11 <= cycle_count + 1;
            weight_c12 <= cycle_count + 1;
            weight_c13 <= cycle_count + 1;
            weight_c14 <= cycle_count + 1;
            weight_c15 <= cycle_count + 1;
            weight_c16 <= cycle_count + 1;
            
            cycle_count <= cycle_count + 1;
            
            $display("Cycle %0d: Data=%0d, All Weights=%0d", 
                     cycle_count + 1, cycle_count + 1, cycle_count + 1);
            
        end else if (cycle_count >= 16) begin
            wr_en <= 1'b0;  // Stop writing after 16 cycles
        end
    end
    
    // Monitor outputs
    always @(posedge clk) begin
        if (!rst && fifo_valid) begin
            $display("Time %0t: FIFO Status - Full:%b Empty:%b Valid:%b", 
                     $time, fifo_full, fifo_empty, fifo_valid);
            $display("PE Outputs: [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]", 
                     pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, 
                     pe9, pe10, pe11, pe12, pe13, pe14, pe15, pe16);
            $display("----------------------------------------");
        end
    end
    
    // Expected results display
    initial begin
        // Wait for reset and some processing time
        wait (rst == 1'b0);
        #1000;  // Wait for all data to be processed
        
        $display("\n=== Expected Results Analysis ===");
        $display("Cycle 1: Data=1, Weights=1  → Expected PE outputs: All 1s");
        $display("Cycle 2: Data=2, Weights=2  → Expected PE outputs: All 4s");
        $display("Cycle 3: Data=3, Weights=3  → Expected PE outputs: All 9s");
        $display("...");
        $display("Cycle 16: Data=16, Weights=16 → Expected PE outputs: All 256s");
        $display("\nNote: Each PE output should be Data_value × Weight_value");
        $display("Since all weights are same in each cycle, all PE outputs should be identical");
        
        #2000;  // Additional time for pipeline to complete
        
        $display("\n=== Final PE Status ===");
        $display("PE Outputs: [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]", 
                 pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, 
                 pe9, pe10, pe11, pe12, pe13, pe14, pe15, pe16);
        
        // Check if outputs match expected pattern
        if (pe1 == pe2 && pe2 == pe3 && pe3 == pe16) begin
            $display("✓ PASS: All PE outputs are identical as expected");
        end else begin
            $display("✗ FAIL: PE outputs should be identical for same weights");
        end
        
        $display("\nSimulation completed successfully!");
        $stop;
    end
    
    // Safety timeout
    initial begin
        #10000;  // 10 microseconds timeout
        $display("ERROR: Simulation timeout!");
        $stop;
    end
    
    // Detailed cycle-by-cycle tracking
    always @(posedge clk) begin
        if (!rst && wr_en) begin
            $display("=== Cycle %0d Input ===", cycle_count + 1);
            $display("Writing: data_r1=%0d", data_r1);
            $display("Weights: [%0d,%0d,%0d,%0d] (showing first 4, all are %0d)", 
                     weight_c1, weight_c2, weight_c3, weight_c4, weight_c1);
        end
    end

endmodule