//
//  ViewController.swift
//  testBrainCore
//
//  Created by Ross on 2017. 8. 31..
//  Copyright © 2017년 wanted. All rights reserved.
//

import UIKit
import BrainCore
import Upsurge
import Metal

class ViewController: UIViewController {
    
    class Source: DataLayer {
        let name: String?
        let id = UUID()
        var data: Blob
        var batchSize: Int
        
        var outputSize: Int {
            return data.count / batchSize
        }
        
        init(name: String, data: Blob, batchSize: Int) {
            self.name = name
            self.data = data
            self.batchSize = batchSize
        }
        
        func nextBatch(_ batchSize: Int) -> Blob {
            return data
        }
    }
    
    class Sink: SinkLayer {
        let name: String?
        let id = UUID()
        var inputSize: Int
        var batchSize: Int
        
        var data: Blob = []
        
        init(name: String, inputSize: Int, batchSize: Int) {
            self.name = name
            self.inputSize = inputSize
            self.batchSize = batchSize
        }
        
        func consume(_ input: Blob) {
            self.data = input
        }
    }


    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        AAA()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    var device: MTLDevice? {
        guard let d = MTLCreateSystemDefaultDevice() else {
            //fatalError("Failed to create a Metal device")
            return nil
        }
        
        return d
    }
    
    var library: MTLLibrary {
        guard let path = Bundle(for: type(of: self)).path(forResource: "default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }
        return try! device!.makeLibrary(filepath: path)
    }

    func AAA() {
        let source = Source(name: "source", data: [1, 1, 2, 2], batchSize: 2)
        let labels = Source(name: "labels", data: [1, 2], batchSize: 2)
        let weights = Matrix<Float>(rows: 2, columns: 1, elements: [2, 4])
        let biases = ValueArray<Float>([1])
        
        let ip = InnerProductLayer(weights: weights, biases: biases, name: "ip")
        let loss = L2LossLayer(size: 1, name: "loss")
        let sink = Sink(name: "sink", inputSize: 1, batchSize: 2)
        
        let net = Net.build({
            source => ip
            [ip, labels] => loss => sink
        })
        
        //let expecation = expectation(description: "Net forward/backward pass")
        var ipInputDiff = [Float]()
        var ipWeightsDiff = [Float]()
        var ipBiasDiff = [Float]()
        
        if nil == device {
            return
        }
        
        let trainer = try! Trainer(net: net, device: device!, batchSize: 2)
        trainer.run() { snapshot in
            ipInputDiff = [Float](snapshot.inputDeltasOfLayer(ip)!)
//            ipWeightsDiff = arrayFromBuffer(ip.weightDeltasBuffer!.metalBuffer!)
//            ipBiasDiff = arrayFromBuffer(ip.biasDeltasBuffer!.metalBuffer!)
//            
            //expecation.fulfill()
        }
        
        //waitForExpectations(timeout: 5) { error in
//            if let error = error {
//                XCTFail("trainer.run() failed: \(error)")
//            }
//            
//            XCTAssertEqual(ipInputDiff, [6, 11, 12, 22])
//            XCTAssertEqual(ipWeightsDiff, [14, 14])
//            XCTAssertEqual(ipBiasDiff, [8.5])
        //}
    }
}

