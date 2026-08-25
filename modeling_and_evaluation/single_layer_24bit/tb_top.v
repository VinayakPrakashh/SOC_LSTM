`timescale 1ns / 1ps

module tb_top_s3_20;

    // ========================================================================
    // TESTBENCH SIGNALS
    // ========================================================================
    reg clk;
    reg rst_n;
    reg start;
    wire done;
    wire [23:0] final_output;
    
    // ========================================================================
    // CLOCK GENERATION (10ns period = 100MHz)
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // ========================================================================
    // DUT INSTANTIATION
    // ========================================================================
    top_s3_20 dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .final_output(final_output)
    );
    
    // ========================================================================
    // TEST STIMULUS
    // ========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        start = 0;
        
        // Reset pulse
        #20;
        rst_n = 1;
        #20;
        
        // Start processing
        $display("========================================");
        $display("Starting LSTM S3.20 Processing");
        $display("Time: %0t", $time);
        $display("========================================");
        
        start = 1;
        #10;
        start = 0;
        
        // Wait for done signal
        wait(done);
        
        $display("========================================");
        $display("Processing Complete!");
        $display("Time: %0t", $time);
        $display("Final Output: 0x%06X (%f)", final_output, $itor(final_output)/(2.0**20));
        $display("========================================");
        
        #100;
        $finish;
    end
    
    // ========================================================================
    // MONITORS
    // ========================================================================
    initial begin
        $monitor("Time=%0t | rst_n=%b | start=%b | done=%b | final_output=0x%06X", 
                 $time, rst_n, start, done, final_output);
    end
    
    // ========================================================================
    // WAVEFORM DUMP
    // ========================================================================
    initial begin
        $dumpfile("tb_top_s3_20.vcd");
        $dumpvars(0, tb_top_s3_20);
    end
    
    // ========================================================================
    // TIMEOUT WATCHDOG
    // ========================================================================
//    initial begin
//        #1000000; // 1ms timeout
//        $display("ERROR: Simulation timeout!");
//        $finish;
//    end

endmodule