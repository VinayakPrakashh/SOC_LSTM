// Simple Testbench for Sigmoid Address Calculator
`timescale 1ns / 1ps

module tb_sigmoid_addr_calc_simple();

    // Test signals
    reg [11:0] data_in;
    wire [8:0] addr_out;
    wire out_of_range;
    
    // DUT instantiation
    sigmoid_addr_calc dut (
        .data_in(data_in),
        .addr_out(addr_out),
        .out_of_range(out_of_range)
    );
    
    // Test procedure
    initial begin
        $display("Sigmoid Address Calculator - Simple Test");
        $display("Input\t\tAddr\tOOR");
        $display("-----\t\t----\t---");
        
        // Test 1: Zero input
        data_in = 12'h000;  // 0.0
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        // Test 2: Small positive
        data_in = 12'b100000100000;  // 0.5
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        // Test 3: Mid range
        data_in = 12'h0C0;  // 3.0
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        // Test 4: Maximum valid
        data_in = 12'h180;  // 6.0
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        // Test 5: Overflow
        data_in = 12'h200;  // 8.0
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        // Test 6: Negative
        data_in = 12'h820;  // -0.5 (sign bit set)
        #10;
        $display("%h\t\t%d\t%b", data_in, addr_out, out_of_range);
        
        $display("Test completed");
        $finish;
    end
    
endmodule