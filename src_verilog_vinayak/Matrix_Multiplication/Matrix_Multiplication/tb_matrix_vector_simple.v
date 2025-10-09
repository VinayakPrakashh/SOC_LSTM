`timescale 1ns / 1ps

module tb_matrix_vector_simple();

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
        forever #5 clk = ~clk;  // 100MHz clock
    end
    
    // Reset sequence
    initial begin
        rst = 1'b1;
        #50;
        rst = 1'b0;
    end
    
    // Simple display task
    task display_results;
        input [7*8:1] test_name;
        begin
            $display("\n=== %s ===", test_name);
            $display("Time: %0t ns", $time);
            $display("Input vector: %d", data_r1);
            $display("Weight matrix (1x16): [%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d]", 
                     weight_c1, weight_c2, weight_c3, weight_c4, weight_c5, weight_c6, weight_c7, weight_c8,
                     weight_c9, weight_c10, weight_c11, weight_c12, weight_c13, weight_c14, weight_c15, weight_c16);
            $display("Output vector (16x1):");
            $display("  [%d %d %d %d]", pe1, pe2, pe3, pe4);
            $display("  [%d %d %d %d]", pe5, pe6, pe7, pe8);
            $display("  [%d %d %d %d]", pe9, pe10, pe11, pe12);
            $display("  [%d %d %d %d]", pe13, pe14, pe15, pe16);
            $display("Expected: All outputs = input * corresponding_weight");
            $display("----------------------------------------");
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Matrix-Vector Multiplication Demonstration");
        $display("==========================================");
        $display("This testbench demonstrates 1x16 weight matrix multiplication with scalar input");
        $display("Output[i] = Input_scalar * Weight[i]");
        
        // Wait for reset to complete
        wait (rst == 1'b0);
        repeat(10) @(posedge clk);
        
        // Test 1: Simple multiplication with input = 2
        $display("\n>>> Test 1: Input = 2, Weights = [1,2,3,4,...,16]");
        @(posedge clk);
        wr_en = 1'b1;
        data_r1 = 12'd2;  // Input scalar
        // Weight matrix: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        weight_c1 = 12'd1;   weight_c2 = 12'd2;   weight_c3 = 12'd3;   weight_c4 = 12'd4;
        weight_c5 = 12'd5;   weight_c6 = 12'd6;   weight_c7 = 12'd7;   weight_c8 = 12'd8;
        weight_c9 = 12'd9;   weight_c10 = 12'd10; weight_c11 = 12'd11; weight_c12 = 12'd12;
        weight_c13 = 12'd13; weight_c14 = 12'd14; weight_c15 = 12'd15; weight_c16 = 12'd16;
        @(posedge clk);
        wr_en = 1'b0;
        
        // Wait for processing
        repeat(25) @(posedge clk);
        display_results("Test 1");
        $display("Expected outputs: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]");
        
        #200;
        
        // Test 2: Simple multiplication with input = 5
        $display("\n>>> Test 2: Input = 5, All weights = 3");
        @(posedge clk);
        wr_en = 1'b1;
        data_r1 = 12'd5;  // Input scalar
        // All weights = 3
        weight_c1 = 12'd3;  weight_c2 = 12'd3;  weight_c3 = 12'd3;  weight_c4 = 12'd3;
        weight_c5 = 12'd3;  weight_c6 = 12'd3;  weight_c7 = 12'd3;  weight_c8 = 12'd3;
        weight_c9 = 12'd3;  weight_c10 = 12'd3; weight_c11 = 12'd3; weight_c12 = 12'd3;
        weight_c13 = 12'd3; weight_c14 = 12'd3; weight_c15 = 12'd3; weight_c16 = 12'd3;
        @(posedge clk);
        wr_en = 1'b0;
        
        repeat(25) @(posedge clk);
        display_results("Test 2");
        $display("Expected outputs: All should be 15 (5 * 3)");
        
        #200;
        
        // Test 3: Input = 10, alternating weights
        $display("\n>>> Test 3: Input = 10, Alternating weights [1,0,1,0,...]");
        @(posedge clk);
        wr_en = 1'b1;
        data_r1 = 12'd10;  // Input scalar
        // Alternating weights: 1,0,1,0,1,0...
        weight_c1 = 12'd1;  weight_c2 = 12'd0;  weight_c3 = 12'd1;  weight_c4 = 12'd0;
        weight_c5 = 12'd1;  weight_c6 = 12'd0;  weight_c7 = 12'd1;  weight_c8 = 12'd0;
        weight_c9 = 12'd1;  weight_c10 = 12'd0; weight_c11 = 12'd1; weight_c12 = 12'd0;
        weight_c13 = 12'd1; weight_c14 = 12'd0; weight_c15 = 12'd1; weight_c16 = 12'd0;
        @(posedge clk);
        wr_en = 1'b0;
        
        repeat(25) @(posedge clk);
        display_results("Test 3");
        $display("Expected outputs: [10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0]");
        
        #200;
        
        // Test 4: Zero input
        $display("\n>>> Test 4: Input = 0, Random weights");
        @(posedge clk);
        wr_en = 1'b1;
        data_r1 = 12'd0;   // Zero input
        weight_c1 = 12'd7;   weight_c2 = 12'd14;  weight_c3 = 12'd21;  weight_c4 = 12'd28;
        weight_c5 = 12'd35;  weight_c6 = 12'd42;  weight_c7 = 12'd49;  weight_c8 = 12'd56;
        weight_c9 = 12'd63;  weight_c10 = 12'd70; weight_c11 = 12'd77; weight_c12 = 12'd84;
        weight_c13 = 12'd91; weight_c14 = 12'd98; weight_c15 = 12'd105; weight_c16 = 12'd112;
        @(posedge clk);
        wr_en = 1'b0;
        
        repeat(25) @(posedge clk);
        display_results("Test 4");
        $display("Expected outputs: All should be 0 (0 * any_weight = 0)");
        
        #200;
        
        // Test 5: Large numbers
        $display("\n>>> Test 5: Input = 100, Small weights");
        @(posedge clk);
        wr_en = 1'b1;
        data_r1 = 12'd100; // Large input
        // Small weights: 0.1 represented as 1 in fixed point
        weight_c1 = 12'd1;  weight_c2 = 12'd1;  weight_c3 = 12'd1;  weight_c4 = 12'd1;
        weight_c5 = 12'd2;  weight_c6 = 12'd2;  weight_c7 = 12'd2;  weight_c8 = 12'd2;
        weight_c9 = 12'd3;  weight_c10 = 12'd3; weight_c11 = 12'd3; weight_c12 = 12'd3;
        weight_c13 = 12'd4; weight_c14 = 12'd4; weight_c15 = 12'd4; weight_c16 = 12'd4;
        @(posedge clk);
        wr_en = 1'b0;
        
        repeat(25) @(posedge clk);
        display_results("Test 5");
        $display("Expected outputs: [100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400]");
        
        // Final summary
        $display("\n==========================================");
        $display("Matrix-Vector Multiplication Demo Complete!");
        $display("==========================================");
        $display("Summary:");
        $display("- The module implements: Output_vector = Input_scalar * Weight_matrix");
        $display("- Input scalar is broadcasted and multiplied with each weight");
        $display("- 16 processing elements compute 16 parallel multiplications");
        $display("- Results appear after pipeline delay (~20-25 clock cycles)");
        $display("==========================================");
        
        $stop;
    end
    
    // Safety timeout
    initial begin
        #50000;
        $display("ERROR: Simulation timeout!");
        $stop;
    end

endmodule